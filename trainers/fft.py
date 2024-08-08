import os

import torch
from transformers import RobertaConfig

from trainers.base import BaseTrainer
from utils.custom import is_rank_0
from utils.modeling_roberta import RobertaForMaskedLM
from utils.xformer import load_base_model


class Trainer(BaseTrainer):
	
	def __init__(self, args, logger):
		super(Trainer, self).__init__(args, logger)
	
	def _build_model(self):
		# Load the Base model
		config, model = load_base_model(
			self.args,
			model_type=self.args.model_type,
			model_name_or_path=self.args.model_name_or_path,
			model_class=RobertaForMaskedLM,  # This is my custom class not the one from transformers
			config_class=RobertaConfig
		)
		return config, model
	
	def init_trackers(self):
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": f"GLUE/{self.args.dataset_name}/fft"}},
			)
	
	def count_parameters(self):
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		all_params = sum(p.numel() for p in self.model.parameters())
		return None, None, trainable_params, all_params
	
	def forward(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		output = self.model(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
			mask_pos=batch['mask_pos'],
			labels=batch['labels']
		)
		return output
	
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
