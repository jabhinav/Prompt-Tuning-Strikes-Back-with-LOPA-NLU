import argparse
import copy
from custom_benchmark.benchmark import PyTorchBenchmark
from custom_benchmark.benchmark_args import PyTorchBenchmarkArguments
from utils.model import LatentPromptAttentionGenerator, IDPGSoftPromptGenerator, IDPGSoftPromptGenerator_wPHM
from utils.xformer import load_base_model, get_huggingface_path
from transformers import RobertaConfig, RobertaModel

# _args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[1], sequence_lengths=[8, 32])
# benchmark = PyTorchBenchmark(_args)
# results = benchmark.run()
# print(results)

do_mem_runtime_benchmark = False

prompt_len = [10]
for t in prompt_len:
	print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
	print(f"==================== Benchmarking for Number of virtual tokens = {t} ====================")
	print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
	parser = argparse.ArgumentParser()
	parser.add_argument('--lp_rank', type=int, default=4)
	parser.add_argument('--total_virtual_tokens', type=int, default=t)
	parser.add_argument('--word_embedding_dim', type=int, default=1024)  # For the Roberta model
	args = parser.parse_args()
	args.cache_dir = None
	args.model_revision = 'main'
	args.token = None
	args.trust_remote_code = False
	args.ignore_mismatched_sizes = False
	args.is_regression = False
	args.is_multi_label = False
	
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
	
	args3 = copy.deepcopy(args)
	args3.enc_model_type = 'roberta-large'
	enc3 = IDPGSoftPromptGenerator_wPHM(
		args3,
		MLP_h=256,
		n=16  # Must be divide MLP_h and embedding_dim = 1024
	)
	
	enc1_name = str(enc1)
	enc2_name = str(enc2)
	enc3_name = str(enc3)
	# Get number of trainable parameters
	print(f"Number of trainable parameters in Prompt Tuning: {args1.word_embedding_dim * args1.total_virtual_tokens / 1e6:.2f}M")
	trainable_params = sum(p.numel() for p in enc1.parameters() if p.requires_grad)
	# Add parameters for the shared soft prompt
	trainable_params += args1.word_embedding_dim * args1.total_virtual_tokens
	print(f"Number of trainable parameters in LatentPromptAttentionGenerator: {trainable_params / 1e6:.2f}M")
	trainable_params = sum(p.numel() for p in enc2.parameters() if p.requires_grad)
	print(f"Number of trainable parameters in IDPGSoftPromptGenerator: {trainable_params / 1e6:.2f}M")
	trainable_params = sum(p.numel() for p in enc3.parameters() if p.requires_grad)
	print(f"Number of trainable parameters in IDPGSoftPromptGenerator_wPHM: {trainable_params / 1e6:.2f}M")

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
