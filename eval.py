import torch
import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import PretrainedConfig, default_data_collator, DataCollatorWithPadding, AutoConfig, \
	AutoModelForSequenceClassification

from utils.config import get_config
from utils.data_processors import task_to_keys
from utils.data import load_data, load_labels
from utils.xformer import load_tokenizer, load_base_model, get_huggingface_path


def process_raw_data(args, logger, raw_datasets, tokenizer, config, model):
	# Preprocessing the datasets
	if args.dataset_name is not None:
		sentence1_key, sentence2_key = task_to_keys[args.dataset_name]
	else:
		# Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
		non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
		if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
			sentence1_key, sentence2_key = "sentence1", "sentence2"
		else:
			if len(non_label_column_names) >= 2:
				sentence1_key, sentence2_key = non_label_column_names[:2]
			else:
				sentence1_key, sentence2_key = non_label_column_names[0], None
	
	# Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = None
	if (
			model.config.label2id != PretrainedConfig(num_labels=args.num_labels).label2id
			and args.dataset_name is not None
			and not args.is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if sorted(label_name_to_id.keys()) == sorted(args.label_list):
			logger.info(
				f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
				"Using it!"
			)
			label_to_id = {i: label_name_to_id[args.label_list[i]] for i in range(args.num_labels)}
		else:
			logger.warning(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(args.label_list)}."
				"\nIgnoring the model labels as a result.",
			)
	elif args.dataset_name is None and not args.is_regression:
		label_to_id = {v: i for i, v in enumerate(args.label_list)}
	
	if label_to_id is not None:
		model.config.label2id = label_to_id
		model.config.id2label = {id: label for label, id in config.label2id.items()}
	elif args.dataset_name is not None and not args.is_regression:
		model.config.label2id = {l: i for i, l in enumerate(args.label_list)}
		model.config.id2label = {id: label for label, id in config.label2id.items()}
	
	padding = "max_length" if args.pad_to_max_length else False
	
	def preprocess_function(examples):
		# Tokenize the texts
		texts = (
			(examples[sentence1_key],) if sentence2_key is None else (
				examples[sentence1_key], examples[sentence2_key])
		)
		result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
		
		if "label" in examples:
			if label_to_id is not None:
				# Map labels to IDs (not necessary for GLUE tasks)
				result["labels"] = [label_to_id[l] for l in examples["label"]]
			else:
				# In all cases, rename the column to labels because the model will expect that.
				result["labels"] = examples["label"]
		return result
	
	processed_datasets = raw_datasets.map(
		preprocess_function,
		batched=True,
		remove_columns=raw_datasets["train"].column_names,
		desc="Running tokenizer on dataset",
	)
	
	return processed_datasets


@torch.no_grad()
def evaluate_on_test(args, logger):
	transformers.logging.set_verbosity_error()
	accelerator = Accelerator()
	
	# Get the tokenizer
	logger.info(f"[INFO] Loading tokenizer from {get_huggingface_path(args.model_type)}")
	tokenizer = load_tokenizer(args, args.model_type, args.tokenizer_name)
	
	# Get the dataset
	logger.info("[INFO] Building dataset...")
	raw_datasets = load_data(args=args)
	# Load the label and its information into self.args
	load_labels(raw_datasets=raw_datasets, args=args)
	
	# Get the model
	logger.info("[INFO] Building model...")
	config, model = load_base_model(
		args,
		model_type=args.model_type,
		model_name_or_path=args.model_name_or_path,
		config_class=AutoConfig,
		model_class=AutoModelForSequenceClassification
	)
	if args.load_base_from_path is not None:
		loaded_state_dict = torch.load(args.load_base_from_path, map_location="cpu")
		loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
		model.load_state_dict(loaded_state_dict, strict=True)
		del loaded_state_dict
		logger.info("[INFO] Loaded model checkpoint from path: {}".format(args.load_base_from_path))
	model.eval()
	
	# Process the raw data
	logger.info("[INFO] Processing raw data...")
	processed_datasets = process_raw_data(args, logger, raw_datasets, tokenizer, config, model)
	if args.pad_to_max_length:
		data_collator = default_data_collator
	else:
		data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
	ds_loader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
	
	# we only wrap data loader to avoid extra memory occupation
	model = model.to(accelerator.device)
	ds_loader = accelerator.prepare(ds_loader)
	
	# Get the testing metric function
	if args.dataset_name is not None:
		metric = evaluate.load("glue", args.dataset_name)
	else:
		metric = evaluate.load("accuracy")
	
	samples_seen = 0
	for step, batch in tqdm(
			enumerate(ds_loader),
			desc=f"Testing",
			unit="batch",
			colour="RED",
			leave=False,
			dynamic_ncols=True,
			smoothing=0.04,
			disable=not accelerator.is_main_process,
	):
		with torch.no_grad():
			outputs = model(**batch)
		
		predictions = outputs.logits.argmax(dim=-1) if not args.is_regression else outputs.logits.squeeze()
		predictions, references = accelerator.gather((predictions, batch["labels"]))
		
		# If we are in a multiprocess environment, the last batch has duplicates
		if accelerator.num_processes > 1:
			if step == len(ds_loader) - 1:
				predictions = predictions[: len(ds_loader.dataset) - samples_seen]
				references = references[: len(ds_loader.dataset) - samples_seen]
			else:
				samples_seen += references.shape[0]
		
		metric.add_batch(
			predictions=predictions,
			references=references,
		)
	
	# Compute the final evaluation metric
	test_metrics = metric.compute()
	for key, metric in test_metrics.items():
		logger.info("  |- Test/{}: {:.6f}".format(key, metric))
		if accelerator.is_main_process:
			print("  |- Test/{}: {:.6f}".format(key, metric))


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	evaluate_on_test(args, logger)


if __name__ == '__main__':
	main()
