import os

from accelerate.logging import MultiProcessAdapter
from peft import get_peft_model, LoraConfig
from transformers import RobertaConfig

from utils.config import get_config
from utils.custom import is_rank_0
from utils.modeling_roberta import RobertaForMaskedLM
from utils.trainer import BaseTrainer
from utils.xformer import load_base_model, LORA_IA3_TARGET_MODULES


class Trainer(BaseTrainer):
	
	def __init__(self, args, logger):
		super(Trainer, self).__init__(args, logger)
	
	def _build_model(self):
		# Load the Base model + Classification Head
		config, model = load_base_model(
			self.args,
			model_type=self.args.model_type,
			model_name_or_path=self.args.model_name_or_path,
			model_class=RobertaForMaskedLM,  # This is my custom class not the one from transformers
			config_class=RobertaConfig
		)
		
		# Get the config
		lora_config = LoraConfig(
			r=16,
			lora_alpha=32,
			target_modules=LORA_IA3_TARGET_MODULES[self.args.model_type]["target_modules_lora"],
			lora_dropout=0.1,
			bias="none",
			# modules_to_save=["classifier"],
		)
		
		# Initialize the model adapters
		model = get_peft_model(model, lora_config)
		
		return config, model
	
	def init_trackers(self):
		# Initialize the trackers
		with self.accelerator.main_process_first():
			self.accelerator.init_trackers(
				project_name=self.args.project_name,
				config=vars(self.args),
				init_kwargs={"wandb": {"name": f"GLUE_{self.args.dataset_name}_lora"}},
			)
	
	def count_parameters(self):
		trainable_params, all_params = self.model.get_nb_trainable_parameters()
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
		model.save_pretrained(
			save_directory=os.path.join(save_at, "LoRA"),
			is_main_process=is_rank_0(),
		)
		
		if is_rank_0():
			print(f"[INFO] (epoch={self.epoch}) Saved the model at:", os.path.join(save_at, "LoRA"))


def main():
	args, logger = get_config()
	logger = MultiProcessAdapter(logger, {})  # An adapter to assist with logging in multiprocess.
	
	trainer = Trainer(args, logger)
	trainer.train_loop()


if __name__ == '__main__':
	# LoRA with Roberta not working deepspeed for some reason. (Could be a library implementation issue.)
	# $ accelerate launch --config_file config_basic_nofp16.yaml tune_lora_baseline.py
	main()
