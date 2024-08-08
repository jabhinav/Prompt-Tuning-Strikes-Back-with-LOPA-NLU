import os

from transformers import RobertaConfig

from custom_peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from trainers.base import BaseTrainer
from utils.custom import is_rank_0
from utils.model import DummyModel
from utils.modeling_roberta import RobertaForMaskedLM
from utils.xformer import load_base_model


class Trainer(BaseTrainer):
	
	def __init__(self, args, logger):
		super(Trainer, self).__init__(args, logger)
	
	def _build_model(self):
		# Load the Base model
		seq_cls_config, seq_classifier = load_base_model(
			self.args,
			model_type=self.args.model_type,
			model_name_or_path=self.args.model_name_or_path,
			model_class=RobertaForMaskedLM,  # This is my custom class not the one from transformers
			config_class=RobertaConfig
		)
		
		# Get the config
		pt_config = PromptTuningConfig(
			task_type=TaskType.MASKED_LM,
			prompt_tuning_init=PromptTuningInit.RANDOM,  # TEXT for text, RANDOM for random
			num_virtual_tokens=self.args.num_virtual_tokens,
		)
		
		# Initialize the model adapters
		seq_classifier = get_peft_model(seq_classifier, pt_config)
		
		self.args.total_virtual_tokens = self.args.num_virtual_tokens * pt_config.num_transformer_submodules
		self.args.word_embedding_dim = pt_config.token_dim
		seq_cls_config.total_virtual_tokens = self.args.total_virtual_tokens
		
		self.logger.info("Building the Sequence Classifier done.")
		
		# Wrap the model with a dummy model (which simply shifts the mask token's pos to the right by num_virtual_tokens)
		seq_classifier = DummyModel(seq_cls_config, seq_classifier)
		
		return seq_cls_config, seq_classifier
	
	def init_trackers(self):
		run_name = self.args.run_name if self.args.run_name is not None else f"GLUE/{self.args.dataset_name}/pt"
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": run_name}},
			)
	
	def count_parameters(self):
		trainable_params, all_params = self.model.seq_classifier.get_nb_trainable_parameters()
		return None, None, trainable_params, all_params
	
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
		
		model = self.accelerator.unwrap_model(self.model)
		
		# Save model
		model.seq_classifier.save_pretrained(
			save_directory=os.path.join(save_at, "PEFT"),
			is_main_process=is_rank_0(),
		)
		
		if is_rank_0():
			print(f"[INFO] (epoch={self.epoch}) Saved the model at:", os.path.join(save_at, "PEFT"))
