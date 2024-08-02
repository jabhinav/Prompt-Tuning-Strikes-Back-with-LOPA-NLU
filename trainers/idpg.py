import os

import torch
from transformers import RobertaConfig

from custom_peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model
from trainers.base import BaseTrainer
from utils.custom import is_rank_0
from utils.model import IDPG, IDPGSoftPromptGenerator, IDPGSoftPromptGenerator_wPHM
from utils.modeling_roberta import RobertaForMaskedLM
from utils.xformer import load_base_model


class Trainer(BaseTrainer):
	
	def __init__(self, args, logger):
		super(Trainer, self).__init__(args, logger)
	
	def _build_model(self):
		# 1) Build the Sequence Classifier (It actually is a mask token classifier) wrapped with PEFT
		seq_cls_config, seq_classifier = load_base_model(
			self.args,
			model_type=self.args.model_type,
			model_name_or_path=self.args.model_name_or_path,
			model_class=RobertaForMaskedLM,  # This is my custom class not the one from transformers
			config_class=RobertaConfig
		)
		peft_config = PromptTuningConfig(
			task_type=TaskType.IDPG_MASKED_LM,
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=self.args.num_virtual_tokens,
		)
		seq_classifier = get_peft_model(seq_classifier, peft_config)  # This will freeze the base model
		
		self.args.total_virtual_tokens = self.args.num_virtual_tokens * peft_config.num_transformer_submodules
		self.args.word_embedding_dim = peft_config.token_dim
		seq_cls_config.total_virtual_tokens = self.args.total_virtual_tokens
		
		self.logger.info("Building the Sequence Classifier done.")
		
		# 2) Build the Latent Prompt Generator
		# # Note 1: If using RobertaModel, some weights in ckpt 'roberta-large' will not be loaded like
		# 			'lm_head.decoder.weight' since LM Head is present is RobertaForCausalLM
		# # Note 2: Also, RobertaModel by default has add_pooling_layer=True which adds a pooling layer on top of the encoder.
		# 			Since ckpt does not have it, it will be init and throw a msg. It is fine since we are not using it.
		if self.args.use_phm_layers:
			soft_prompt_gen = IDPGSoftPromptGenerator_wPHM(self.args, n=self.args.phm_n)
		else:
			soft_prompt_gen = IDPGSoftPromptGenerator(self.args)
		
		self.logger.info("Building the Soft Prompt Generator done.")
		
		# 3) Build the IDPG model
		model = IDPG(seq_cls_config, soft_prompt_gen, seq_classifier)
		return seq_cls_config, model
	
	def init_trackers(self):
		# Initialize the trackers
		run_name = self.args.run_name if self.args.run_name is not None else f"GLUE/{self.args.dataset_name}/idpg"
		if self.args.use_phm_layers:
			run_name += f"_PHM_{self.args.phm_n}"
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": run_name}},
			)
	
	def count_parameters(self):
		lp_gen_trainable_params = sum(p.numel() for p in self.model.latent_prompt_gen.parameters() if p.requires_grad)
		lp_gen_all_params = sum(p.numel() for p in self.model.latent_prompt_gen.parameters())
		seq_cls_trainable_params, seq_cls_all_params = self.model.seq_classifier.get_nb_trainable_parameters()
		return lp_gen_trainable_params, lp_gen_all_params, seq_cls_trainable_params, seq_cls_all_params
	
	def forward(self, batch):
		r"""Forward step for training and inference. This function is called
		in ``_train_step`` & ``_test_step`` function.
		"""
		output = self.model(batch)
		return output
	
	def save(self, dir_tag: str):
		
		# Create a directory to save the model
		save_at = os.path.join(self.args.log_dir, dir_tag)
		if not os.path.exists(save_at):
			os.makedirs(save_at)
		
		# Unwrap the model
		model: IDPG = self.accelerator.unwrap_model(self.model)
		
		# Save the latent prompt generator
		state_dict = model.latent_prompt_gen.state_dict()
		# Remove the enc.base layers before saving [IDPG-specific]
		state_dict = {k: v for k, v in state_dict.items() if "base" not in k}
		torch.save(state_dict, os.path.join(save_at, "lp_generator.pt"))
		del state_dict
		
		# Save Sequence Classifier in steps
		
		# First save the PEFT part
		model.seq_classifier.save_pretrained(
			save_directory=os.path.join(save_at, "PEFT"),
			is_main_process=is_rank_0(),
		)
		
		# # Secondly save the classifier head
		# head_state_dict = model.seq_classifier.classifier.state_dict()
		# torch.save(head_state_dict, os.path.join(save_at, "classifier_head.pt"))
		# del head_state_dict
		
		if is_rank_0():
			print(f"[INFO] (epoch={self.epoch}) Saved the classification model at:", os.path.join(save_at, "PEFT"))
			# print(f"[INFO] (epoch={self.epoch}) Saved the classifier head at:", os.path.join(save_at, "classifier_head.pt"))
			print(f"[INFO] (epoch={self.epoch}) Saved the latent prompt encoder at:",
				  os.path.join(save_at, "lp_generator.pt"))
