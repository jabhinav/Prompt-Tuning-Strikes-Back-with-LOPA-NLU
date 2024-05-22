import json
import math
import os
import random
import time
from collections import defaultdict

import accelerate
import datasets
import evaluate
import torch
import transformers
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler, PretrainedConfig, default_data_collator, \
	DataCollatorWithPadding, AutoConfig, AutoModelForSequenceClassification

from utils.config import get_config
from utils.data_processors import task_to_keys
from utils.custom import is_rank_0
from utils.data import load_labels, load_data
from utils.xformer import load_base_model, load_tokenizer, get_huggingface_path


class Trainer(object):
	
	def __init__(self, args, logger):
		self.args = args
		
		# init with accelerate
		self._init_accelerator()
		self.accelerator.wait_for_everyone()
		
		with self.accelerator.main_process_first():
			self.logger = logger
		
		self.logger.info("Accelerator State:\n")
		self.logger.info(self.accelerator.state, main_process_only=False)
		if self.accelerator.is_local_main_process:
			datasets.utils.logging.set_verbosity_warning()
			transformers.utils.logging.set_verbosity_info()
		else:
			datasets.utils.logging.set_verbosity_error()
			transformers.utils.logging.set_verbosity_error()
		
		# Log some info
		self.logger.info("=" * 56)
		self.logger.info("||\t\t" + "New training process started." + "\t\t||")
		self.logger.info("=" * 56)
		self.logger.info("\n")
		self.logger.info(f"Experiment name: {args.project_name}")
		self.logger.info(f"Experiment directory: {self.args.log_dir}")
		
		# init counts
		self.step: int = 0
		self.epoch: int = 0
		
		# setup tokenizer
		logger.info(f"[INFO] Loading tokenizer from {get_huggingface_path(args.model_type)}")
		self.tokenizer = load_tokenizer(args, args.model_type, args.tokenizer_name)
		
		# setup data
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Building dataset...")
			start = time.monotonic_ns()
			self.raw_datasets = self.get_data()
			end = time.monotonic_ns()
			self.logger.info(f"[INFO] Building dataset done in {(end - start) / 1e6:.2f}ms")
		
		# setup model
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Building model...")
			start = time.monotonic_ns()
			self.config, self.model = self._build_model()
			end = time.monotonic_ns()
			self.logger.debug(self.model)
			self.logger.info(f"[INFO] Building model done in {(end - start) / 1e6:.2f}ms")
			
			# Print the number of trainable parameters
			trainable_params, all_params = self.__count_parameters(
				self.model)
			msg = (f"Encoder: trainable params: {trainable_params:,d} || all params: {all_params:,d} ||"
				   f" trainable%: {100 * trainable_params / all_params}")
			self.logger.info(msg)
			
		# Process raw data
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Processing raw data...")
			start = time.monotonic_ns()
			self.processed_datasets = self.process_raw_data()
			self.train_dataset = self.processed_datasets["train"]
			self.eval_dataset = self.processed_datasets["validation_matched" if self.args.dataset_name == "mnli" else "validation"]
			# Log a few random samples from the training set:
			for index in random.sample(range(len(self.train_dataset)), 3):
				self.logger.info(f"Sample {index} of the training set: {self.train_dataset[index]}.")
			end = time.monotonic_ns()
			self.logger.info(f"[INFO] Processing raw data done in {(end - start) / 1e6:.2f}ms")
	
		# Load data Loaders
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Building dataloader...")
			start = time.monotonic_ns()
			self.train_dataloader, self.eval_dataloader, self.data_collator = self._build_dataloader()
			end = time.monotonic_ns()
			self.logger.info(f"[INFO] Building dataloader done in {(end - start) / 1e6:.2f}ms")

		
		# optimizer & scheduler
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Building optimizer and scheduler...")
			start = time.monotonic_ns()
			self.optimizer = self._build_optimizer()
			self.scheduler = self._build_scheduler()
			end = time.monotonic_ns()
			self.logger.info(
				f"[INFO] Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
			)
		
		# accelerate prepare
		self.logger.info("[INFO] Initializing accelerate...")
		start = time.monotonic_ns()
		self._accelerator_prepare()
		end = time.monotonic_ns()
		self.logger.info(f"[INFO] Initializing accelerate done in {(end - start) / 1e6:.2f}ms")
		
		# We need to recalculate our total training steps as the size of the training dataloader may have changed after
		# Accelerator's prepare function.
		self.recalculate_training_metrics()
		
		# Get the metric function
		if self.args.dataset_name is not None:
			self.metric = evaluate.load("glue", args.dataset_name)
		else:
			self.metric = evaluate.load("accuracy")
		
		# save config file path
		self.config_save_path = os.path.join(self.args.log_dir, "args.json")
		self.args.device = self.accelerator.device
		
		# Finally, initialize the trackers. During init of the model we computed new arguments. Thus setting after that.
		self.init_trackers()
	
	def _init_accelerator(self):
		
		project_config = ProjectConfiguration(
			logging_dir=self.args.log_dir,
		)
		kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
		
		# when using DeepSpeed, the `gradient_accumulation_steps` is properly set either
		# > from the DeepSpeed plugin/config
		# > from `accelerate launch` via `--gradient_accumulation_steps`
		# > defaulting to the passed `args.gradient_accumulation_steps` (using this + setting auto in the config file)
		if self.args.wandb_logging:
			self.accelerator = accelerate.Accelerator(
				gradient_accumulation_steps=self.args.gradient_accumulation_steps,
				log_with=["wandb"],
				project_config=project_config,
				kwargs_handlers=[kwargs],
			)
		else:
			self.accelerator = accelerate.Accelerator(
				gradient_accumulation_steps=self.args.gradient_accumulation_steps,
				project_config=project_config,
				kwargs_handlers=[kwargs],
			)
		
	
	def get_data(self):
		raw_data = load_data(args=self.args)
		# Load the label and its information into self.args
		load_labels(raw_datasets=raw_data, args=self.args)
		return raw_data
	
	def process_raw_data(self):
		# Preprocessing the datasets
		if self.args.dataset_name is not None:
			sentence1_key, sentence2_key = task_to_keys[self.args.dataset_name]
		else:
			# Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
			non_label_column_names = [name for name in self.raw_datasets["train"].column_names if name != "label"]
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
				self.model.config.label2id != PretrainedConfig(num_labels=self.args.num_labels).label2id
				and self.args.dataset_name is not None
				and not self.args.is_regression
		):
			# Some have all caps in their config, some don't.
			label_name_to_id = {k.lower(): v for k, v in self.model.config.label2id.items()}
			if sorted(label_name_to_id.keys()) == sorted(self.args.label_list):
				self.logger.info(
					f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
					"Using it!"
				)
				label_to_id = {i: label_name_to_id[self.args.label_list[i]] for i in range(self.args.num_labels)}
			else:
				self.logger.warning(
					"Your model seems to have been trained with labels, but they don't match the dataset: ",
					f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(self.args.label_list)}."
					"\nIgnoring the model labels as a result.",
				)
		elif self.args.dataset_name is None and not self.args.is_regression:
			label_to_id = {v: i for i, v in enumerate(self.args.label_list)}
		
		if label_to_id is not None:
			self.model.config.label2id = label_to_id
			self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}
		elif self.args.dataset_name is not None and not self.args.is_regression:
			self.model.config.label2id = {l: i for i, l in enumerate(self.args.label_list)}
			self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}
		
		padding = "max_length" if self.args.pad_to_max_length else False
		
		def preprocess_function(examples):
			# Tokenize the texts
			texts = (
				(examples[sentence1_key],) if sentence2_key is None else (
				examples[sentence1_key], examples[sentence2_key])
			)
			result = self.tokenizer(*texts, padding=padding, max_length=self.args.max_length, truncation=True)
			
			if "label" in examples:
				if label_to_id is not None:
					# Map labels to IDs (not necessary for GLUE tasks)
					result["labels"] = [label_to_id[l] for l in examples["label"]]
				else:
					# In all cases, rename the column to labels because the model will expect that.
					result["labels"] = examples["label"]
			return result
		
		with self.accelerator.main_process_first():
			processed_datasets = self.raw_datasets.map(
				preprocess_function,
				batched=True,
				remove_columns=self.raw_datasets["train"].column_names,
				desc="Running tokenizer on dataset",
			)
			
		return processed_datasets
	
	
	def _build_dataloader(self):
		# DataLoaders creation:
		if self.args.pad_to_max_length:
			# If padding was already done to max length, we use the default data collator that will just convert everything
			# to tensors.
			data_collator = default_data_collator
		else:
			# Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
			# the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
			# of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
			data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None))
		
		train_dataloader = DataLoader(
			self.train_dataset, shuffle=True, collate_fn=data_collator, batch_size=self.args.per_device_train_batch_size
		)
		eval_dataloader = DataLoader(
			self.eval_dataset, collate_fn=data_collator, batch_size=self.args.per_device_eval_batch_size
		)
		
		num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.accelerator.gradient_accumulation_steps)
		self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch
		
		return train_dataloader, eval_dataloader, data_collator
	
	def _build_model(self):
		# Load the Base model + Classification Head
		config, model = load_base_model(
			self.args,
			model_type=self.args.model_type,
			model_name_or_path=self.args.model_name_or_path,
			config_class=AutoConfig,
			model_class=AutoModelForSequenceClassification
		)
		return config, model
	
	def _build_optimizer(self):
		
		# Split weights in two groups, one with weight decay and the other not.
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": self.args.weight_decay,
			},
			{
				"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
				"weight_decay": 0.0,
			},
		]
		
		# Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
		optimizer_cls = (
			torch.optim.AdamW
			if self.accelerator.state.deepspeed_plugin is None
			   or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
			else accelerate.utils.DummyOptim
		)
		optimizer = optimizer_cls(optimizer_grouped_parameters, lr=self.args.lr)
		
		return optimizer
	
	def _build_scheduler(self):
		
		# Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
		if (
				self.accelerator.state.deepspeed_plugin is None
				or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
		):
			lr_scheduler = get_scheduler(
				name=self.args.lr_scheduler_type,
				optimizer=self.optimizer,
				num_warmup_steps=int(0.06 * self.args.max_train_steps),
				num_training_steps=self.args.max_train_steps,
			)
		else:
			lr_scheduler = accelerate.utils.DummyScheduler(
				self.optimizer,
				total_num_steps=self.args.max_train_steps,
				warmup_num_steps=int(0.06 * self.args.max_train_steps),
			)
		return lr_scheduler
	
	def _accelerator_prepare(self):
		
		self.train_dataloader, self.eval_dataloader, self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
			self.train_dataloader, self.eval_dataloader, self.model, self.optimizer, self.scheduler)
	
	def recalculate_training_metrics(self):
		
		num_update_steps_per_epoch = math.ceil(
			len(self.train_dataloader) / self.accelerator.gradient_accumulation_steps)
		self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch
		
		# # After wards we recalculate our number of training epochs.
		# Keep this. Useful when max_train_steps is to be set manually
		self.args.num_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
		self.args.total_batch_size = (
				self.args.per_device_train_batch_size * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
		)
		
		self.logger.info("\n")
		self.logger.info(f"  Num examples = {len(self.train_dataset)}")
		self.logger.info(f"  Num Epochs = {self.args.num_epochs}")
		self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
		self.logger.info(
			f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args.total_batch_size}")
		self.logger.info(f"  Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}")
		self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
		self.logger.info("\n")
	
	def init_trackers(self):
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": f"GLUE_{self.args.dataset_name}"}},
			)
	
	def _compute_grad_norm(self):
		# Compute the gradients norm
		total_norm = torch.tensor(0.0).to(self.accelerator.device)
		for p in self.model.parameters():
			if p.grad is not None:
				param_norm = p.grad.data.norm(2) ** 2
				total_norm += param_norm.item()
		# Sum gradients across all processes
		total_norm = self.accelerator.reduce(total_norm, reduction="sum")
		total_norm = total_norm ** (1. / 2)
		return total_norm
	
	@staticmethod
	def __count_parameters(model):
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		all_params = sum(p.numel() for p in model.parameters())
		return trainable_params, all_params
	
	def _train_step(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		
		with self.accelerator.accumulate(self.model):
			output = self.model(**batch)
			
			# Classification function = -log(p(x|z))
			clf_loss = output.loss
			
			# Total loss
			total_loss = clf_loss
			
			# BP and Grad Updated
			self.accelerator.backward(total_loss)
			self.optimizer.step()
			total_norm = self._compute_grad_norm()  # compute before optimizer.zero_grad()
			self.optimizer.zero_grad()
			
			if self.accelerator.sync_gradients:
				# Updating the current step under the accumulate context manager takes care of everything
				self.step += 1
		
		return {
			f"total_loss": total_loss.detach().cpu().numpy().item(),
			f"clf_loss": clf_loss.detach().cpu().numpy().item(),
			f"grad_norm": total_norm.detach().cpu().numpy().item(),
		}
	
	def _train_epoch(self):
		r"""Training epoch. Should return average loss of a batch (sample) over
		        one epoch. See ``train_loop`` for usage.
		"""
		
		# Set the model to train mode
		self.model.train()
		
		train_metrics: dict = {}
		
		for batch in tqdm(
				self.train_dataloader,
				desc=f"Training Epoch {self.epoch}",
				unit="batch",
				colour="GREEN",
				leave=False,
				dynamic_ncols=True,
				smoothing=0.04,
				disable=not self.accelerator.is_main_process,
		):
			train_losses = self._train_step(batch)
			
			for key, value in train_losses.items():
				if key not in train_metrics.keys():
					train_metrics[key] = value
				else:
					train_metrics[key] += value
			
			if self.args.wandb_logging:
				self.accelerator.log(
					{
						"Step/Total Loss": train_losses["total_loss"],
						"Step/Classification Loss": train_losses["clf_loss"],
						"Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
						"Step/Gradient Norm": train_losses["grad_norm"],
					},
					step=self.step,
				)
				
			# break
				
		self.accelerator.wait_for_everyone()
		
		# Compute the average losses for the epoch
		for key in train_metrics.keys():
			train_metrics[key] = (
					train_metrics[key] / len(self.train_dataloader) * self.args.gradient_accumulation_steps
			)
		
		return train_metrics
	
	def _eval_epoch(self, dataloader):
		self.model.eval()
		
		samples_seen = 0
		for step, batch in tqdm(
				enumerate(dataloader),
				desc=f"Evaluating Epoch {self.epoch}",
				unit="batch",
				colour="YELLOW",
				leave=False,
				dynamic_ncols=True,
				smoothing=0.04,
				disable=not self.accelerator.is_main_process,
		):
			with torch.no_grad():
				outputs = self.model(**batch)
			
			predictions = outputs.logits.argmax(dim=-1) if not self.args.is_regression else outputs.logits.squeeze()
			predictions, references = self.accelerator.gather((predictions, batch["labels"]))
			
			# If we are in a multiprocess environment, the last batch has duplicates
			if self.accelerator.num_processes > 1:
				if step == len(dataloader) - 1:
					predictions = predictions[: len(dataloader.dataset) - samples_seen]
					references = references[: len(dataloader.dataset) - samples_seen]
				else:
					samples_seen += references.shape[0]
			
			self.metric.add_batch(
				predictions=predictions,
				references=references,
			)
			
		# Compute the final evaluation metric
		eval_metric = self.metric.compute()
		return eval_metric
	
	def train_loop(self):
		r"""Training loop. The public entry of training process."""
		
		best_metrics = defaultdict(lambda: 0)
		
		self.accelerator.wait_for_everyone()
		while self.epoch < self.args.num_epochs:
			self.logger.info("\n")
			self.logger.info("-" * 32)
			self.logger.info("Epoch {}: ".format(self.epoch))
			
			# Do training epoch
			train_metrics = self._train_epoch()
			
			# Do evaluation epoch
			eval_metrics = self._eval_epoch(dataloader=self.eval_dataloader)
			
			# Log the metrics
			for key, metric in train_metrics.items():
				self.logger.info("  |- Train/{}: {:.6f}".format(key, metric))
				if self.args.wandb_logging:
					self.accelerator.log({"Epoch/{}".format(key): metric}, step=self.step)
			
			for key, metric in eval_metrics.items():
				self.logger.info("  |- Eval/{}: {:.6f}".format(key, metric))
				if self.args.wandb_logging:
					self.accelerator.log({"Epoch/{}".format(key): metric}, step=self.step)
					
				# Tracking the best metrics
				if key in best_metrics.keys():
					if metric > best_metrics[key]:
						best_metrics[key] = metric
						self.accelerator.wait_for_everyone()
						if self.accelerator.is_main_process:
							self.save(f"best_{key}")
				else:
					best_metrics[key] = metric
					self.accelerator.wait_for_everyone()
					if self.accelerator.is_main_process:
						self.save(f"best_{key}")
					
			# Update info for each epoch
			self.epoch += 1
			
			if self.args.save_every > 0 and self.epoch % self.args.save_every == 0:
				self.accelerator.wait_for_everyone()
				if self.accelerator.is_main_process:
					self.save(f"epoch_{self.epoch}")
		
		# Finish training and save final checkpoint
		self.accelerator.wait_for_everyone()
		if self.accelerator.is_main_process:
			# self.accelerator.save_state(os.path.join(self.args.log_dir, "final_epoch"))
			self.save("final")
		
		self.accelerator.end_training()
		
		if self.args.dataset_name == "mnli":
			# Final evaluation on mismatched validation set
			eval_dataloader = DataLoader(
				self.eval_dataset, collate_fn=self.data_collator, batch_size=self.args.per_device_eval_batch_size
			)
			eval_dataloader = self.accelerator.prepare(eval_dataloader)
			
			self.model.eval()
			for step, batch in enumerate(eval_dataloader):
				outputs = self.model(**batch)
				predictions = outputs.logits.argmax(dim=-1)
				self.metric.add_batch(
					predictions=self.accelerator.gather(predictions),
					references=self.accelerator.gather(batch["labels"]),
				)
			
			eval_metrics = self.metric.compute()
			for key, metric in eval_metrics.items():
				self.logger.info("  |- (mnli-mm) Final Eval/{}: {:.6f}".format(key, metric))
	
	
	def save(self, dir_tag: str):
		
		# Create a directory to save the model
		save_at = os.path.join(self.args.log_dir, dir_tag)
		if not os.path.exists(save_at):
			os.makedirs(save_at)
		
		model = self.accelerator.unwrap_model(self.model)
		
		# Save model
		torch.save(model.state_dict(), os.path.join(save_at, "nlu_model.pt"))
		
		if is_rank_0():
			print(f"[INFO] (epoch={self.epoch}) Saved the model at:", os.path.join(save_at, "nlu_model.pt"))
	
		
def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	trainer = Trainer(args, logger)
	trainer.train_loop()


if __name__ == '__main__':
	# $ accelerate launch --config_file config_ds_zero_stage2_no_fp16.yaml train.py
	main()
