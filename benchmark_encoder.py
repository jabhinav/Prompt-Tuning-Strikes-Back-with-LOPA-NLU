import argparse
import copy
from custom_benchmark.benchmark import PyTorchBenchmark
from custom_benchmark.benchmark_args import PyTorchBenchmarkArguments
from utils.model import LatentPromptAttentionGenerator, IDPGSoftPromptGenerator
from utils.xformer import load_base_model, get_huggingface_path
from transformers import RobertaConfig, RobertaModel

# _args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[1], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(_args)
# results = benchmark.run()
# print(results)

do_mem_runtime_benchmark = False

prompt_len = [100, 50, 25, 10, 5]
for t in prompt_len:
	print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
	print(f"==================== Benchmarking for Number of virtual tokens = {t} ====================")
	print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
	parser = argparse.ArgumentParser()
	parser.add_argument('--lp_rank', type=int, default=1)
	parser.add_argument('--total_virtual_tokens', type=int, default=t)
	parser.add_argument('--word_embedding_dim', type=int, default=1024)  # For the Roberta model
	args = parser.parse_args()

	
	args1 = copy.deepcopy(args)
	args1.enc_model_type = "roberta-large"
	enc1 = LatentPromptAttentionGenerator(
		args1,
		freeze_base=True,
		MLP_h=256
	)
	
	args2 = copy.deepcopy(args)
	args2.enc_model_type = 'roberta-large'
	enc2 = IDPGSoftPromptGenerator(
		args2,
		MLP_h=256
	)
	
	enc1_name = str(enc1)
	enc2_name = str(enc2)
	# Get number of trainable parameters
	print(f"Number of trainable parameters in Prompt Tuning: {args1.word_embedding_dim * args1.total_virtual_tokens / 1e6:.2f}M")
	trainable_params = sum(p.numel() for p in enc1.parameters() if p.requires_grad)
	# Add parameters for the shared soft prompt
	trainable_params += args1.word_embedding_dim * args1.total_virtual_tokens
	print(f"Number of trainable parameters in LatentPromptAttentionGenerator: {trainable_params / 1e6:.2f}M")
	trainable_params = sum(p.numel() for p in enc2.parameters() if p.requires_grad)
	print(f"Number of trainable parameters in IDPGSoftPromptGenerator: {trainable_params / 1e6:.2f}M")

	if do_mem_runtime_benchmark:
		_args = PyTorchBenchmarkArguments(
			load_my_custom_model=True,
			models=[enc1_name, enc2_name],
			custom_models={
				enc1_name: enc1,
				enc2_name: enc2,
			},
			batch_sizes=[1],
			sequence_lengths=[325],
		)
		
		print("Models being benchmarked: ", _args.models)
		
		benchmark = PyTorchBenchmark(_args, configs=[enc1.config, enc2.config])
		results = benchmark.run()
		print(results)
