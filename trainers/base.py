import math
import os
import time
from collections import defaultdict

import accelerate
import datasets
import evaluate
import torch
import transformers
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import get_scheduler

from utils.data import PromptDataset_wEnc as PromptDataset
from utils.data_processors import processors, output_modes, task_mappings_for_eval
from utils.xformer import load_tokenizer, get_huggingface_path


class BaseTrainer(object):
	
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
		logger.info(f"[INFO] Loading Sequence Classifier's tokenizer from {get_huggingface_path(args.model_type)}")
		self.tokenizer = load_tokenizer(args, args.model_type, args.tokenizer_name)
		logger.info(f"[INFO] Loading Prompt Generator's tokenizer from {get_huggingface_path(args.lp_gen_model_type)}")
		self.lp_gen_tokenizer = load_tokenizer(args, args.lp_gen_model_type,
											   get_huggingface_path(args.lp_gen_model_type))
		
		# prepare glue task
		self.prepare_glue_task()
		
		# setup model
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Building model...")
			start = time.monotonic_ns()
			self.config, self.model = self._build_model()
			end = time.monotonic_ns()
			self.logger.debug(self.model)
			self.logger.info(f"[INFO] Building model done in {(end - start) / 1e6:.2f}ms")
			
			# Get the number of trainable parameters
			lp_gen_trainable_params, lp_gen_all_params, seq_cls_trainable_params, seq_cls_all_params = self.count_parameters()
			if lp_gen_all_params is not None and lp_gen_trainable_params is not None:
				msg = (
					f"Soft (Latent) Prompt Generator: trainable params: {lp_gen_trainable_params:,d} || all params: {lp_gen_all_params:,d} ||"
					f" trainable%: {100 * lp_gen_trainable_params / lp_gen_all_params}")
				self.logger.info(msg)
			if seq_cls_all_params is not None and seq_cls_trainable_params is not None:
				msg = (
					f"Sequence Classifier: trainable params: {seq_cls_trainable_params:,d} || all params: {seq_cls_all_params:,d} ||"
					f" trainable%: {100 * seq_cls_trainable_params / seq_cls_all_params}")
				self.logger.info(msg)
		
		# Get data
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Building dataset...")
			start = time.monotonic_ns()
			self.train_dataset, self.eval_dataset = self.get_data()
			end = time.monotonic_ns()
			self.logger.info(f"[INFO] Building dataset done in {(end - start) / 1e6:.2f}ms")
		
		# Load data Loaders
		with self.accelerator.main_process_first():
			self.logger.info("[INFO] Building dataloader...")
			start = time.monotonic_ns()
			self.train_dataloader, self.eval_dataloader = self._build_dataloader()
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
		
		# Setup the evaluation
		self.label_ids = self.setup_eval()
		# Get the metric function
		if self.args.dataset_name is not None:
			self.metric = evaluate.load("glue", task_mappings_for_eval[self.args.dataset_name])
		else:
			self.metric = evaluate.load("accuracy")
		
		# save config file path
		self.config_save_path = os.path.join(self.args.log_dir, "args.json")
		self.args.device = self.accelerator.device
		
		# Finally, initialize the trackers. During init of the model we computed new arguments. Thus setting after that.
		self.init_trackers()
	
	def prepare_glue_task(self):
		task_name = self.args.dataset_name
		processor = processors[task_name]()
		self.args.output_mode = output_modes[task_name]
		self.args.is_regression = self.args.output_mode == "regression"
		self.args.is_multi_label = self.args.output_mode == "multilabel_classification"
		self.args.label_list = processor.get_labels()
		self.args.num_labels = len(self.args.label_list)
	
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
		train_dataset = PromptDataset(self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer,
									  data_type='train')
		eval_dataset = PromptDataset(self.args, self.args.dataset_name, self.tokenizer, self.lp_gen_tokenizer,
									 data_type='dev')
		return train_dataset, eval_dataset
	
	def _build_dataloader(self):
		train_sampler = RandomSampler(self.train_dataset) if self.args.local_rank == -1 else DistributedSampler(
			self.train_dataset)
		train_dataloader = DataLoader(
			self.train_dataset,
			sampler=train_sampler,
			batch_size=self.args.per_device_train_batch_size,
			collate_fn=self.train_dataset.collate_fn
		)
		
		eval_sampler = SequentialSampler(self.eval_dataset)
		eval_dataloader = DataLoader(
			self.eval_dataset,
			sampler=eval_sampler,
			batch_size=self.args.per_device_eval_batch_size,
			collate_fn=self.eval_dataset.collate_fn
		)
		
		num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.accelerator.gradient_accumulation_steps)
		self.args.max_train_steps = self.args.num_epochs * num_update_steps_per_epoch
		
		return train_dataloader, eval_dataloader
	
	def _build_model(self):
		raise NotImplementedError
	
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
		raise NotImplementedError
	
	def count_parameters(self):
		raise NotImplementedError
	
	def forward(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		raise NotImplementedError
	
	def _train_step(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		
		with self.accelerator.accumulate(self.model):
			output = self.forward(batch)
			
			# Classification function = -log(p(x|z))
			clf_loss = output.loss
			
			# Total loss
			total_loss = clf_loss
			
			# BP and Grad Updated
			self.accelerator.backward(total_loss)
			# Compute the gradients norm
			self.optimizer.step()
			self.scheduler.step()
			self.optimizer.zero_grad()
			
			if self.accelerator.sync_gradients:
				# Updating the current step under the accumulate context manager takes care of everything
				self.step += 1
		
		return {
			f"total_loss": total_loss.detach().cpu().numpy().item(),
			f"clf_loss": clf_loss.detach().cpu().numpy().item(),
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
					},
					step=self.step,
				)
		
		self.accelerator.wait_for_everyone()
		
		# Compute the average losses for the epoch
		for key in train_metrics.keys():
			train_metrics[key] = (
					train_metrics[key] / len(self.train_dataloader) * self.args.gradient_accumulation_steps
			)
		
		return train_metrics
	
	def setup_eval(self):
		processor = processors[self.args.dataset_name]()
		label_ids = []
		label_map = processor.get_label_map()
		for k, v in label_map.items():
			label_id = self.tokenizer(' ' + v, add_special_tokens=False)['input_ids']
			assert len(label_id) == 1
			label_ids.append(label_id[0])
		return label_ids
	
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
				total=len(dataloader),
		):
			with torch.no_grad():
				outputs = self.forward(batch)
				logits = outputs.logits
				# Logits for label ids
				logits = logits[:, self.label_ids]
			
			logits, references = self.accelerator.gather((logits, batch["labels"]))
			
			# Update the label ids in the references to 0, 1, ...
			for i, label in enumerate(self.label_ids):
				references[references == label] = i
			# Get the predictions
			predictions = logits.argmax(dim=-1) if not self.args.is_regression else logits.squeeze()
			
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
		# Do evaluation epoch at the start
		self.logger.info("\n")
		self.logger.info("-" * 32)
		self.logger.info("Epoch {}: ".format(-1))
		eval_metrics = self._eval_epoch(dataloader=self.eval_dataloader)
		
		best_overall = sum(eval_metrics.values())/len(eval_metrics)
		for key, metric in eval_metrics.items():
			best_metrics[key] = metric
			self.logger.info("  |- Eval/{}: {:.6f}".format(key, metric))
		
		self.accelerator.wait_for_everyone()
		while self.epoch < self.args.num_epochs:
			self.logger.info("\n")
			self.logger.info("-" * 32)
			self.logger.info("Epoch {}: ".format(self.epoch))
			
			# Do training epoch
			train_metrics = self._train_epoch()
			
			# Do evaluation epoch
			eval_metrics = self._eval_epoch(dataloader=self.eval_dataloader)
			
			# Update the best overall
			overall = sum(eval_metrics.values())/len(eval_metrics)
			if overall > best_overall:
				best_overall = overall
			
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
		
		self.logger.info("\n")
		self.logger.info("=" * 32)
		self.logger.info("Training done.")
		self.logger.info("=" * 32)
		self.logger.info("Best overall performance so far : {:.6f}".format(best_overall))
	
	# if self.args.dataset_name == "mnli":
	# 	# Final evaluation on mismatched validation set
	# 	eval_dataloader = DataLoader(
	# 		self.eval_dataset, collate_fn=self.data_collator, batch_size=self.args.per_device_eval_batch_size
	# 	)
	# 	eval_dataloader = self.accelerator.prepare(eval_dataloader)
	#
	# 	self.model.eval()
	# 	for step, batch in enumerate(eval_dataloader):
	# 		outputs = self.model(batch)
	# 		predictions = outputs.logits.argmax(dim=-1)
	# 		self.metric.add_batch(
	# 			predictions=self.accelerator.gather(predictions),
	# 			references=self.accelerator.gather(batch["labels"]),
	# 		)
	#
	# 	eval_metrics = self.metric.compute()
	# 	for key, metric in eval_metrics.items():
	# 		self.logger.info("  |- (mnli-mm) Final Eval/{}: {:.6f}".format(key, metric))
	
	def save(self, dir_tag: str):
		
		raise NotImplementedError
