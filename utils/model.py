from typing import Optional

import math
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput

from custom_peft import PeftIDPGModelForMaskedLM, PeftModelForMaskedLM, PeftLopaModelForMaskedLM
from utils.xformer import load_base_model, get_huggingface_path
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F


class LatentPromptAttentionGenerator(torch.nn.Module):
	"""
	Used in LOPA to generate the instance-specific attention weights, Z_I before the gating function.
	"""
	def __init__(self, args, use_bias=True, freeze_base=False, MLP_h=256):
		super(LatentPromptAttentionGenerator, self).__init__()
		
		config, base = load_base_model(
			args,
			model_type=args.enc_model_type,
			model_name_or_path=get_huggingface_path(args.enc_model_type)
		)
		
		self.args = args
		self.config = config
		self.base = base  # CodeBERT model
		self.rank = self.args.lp_rank
		
		# # Base model does not require any training - freeze the weights
		self.freeze_base = freeze_base
		if self.freeze_base:
			for param in self.base.parameters():
				param.requires_grad = False
		
		# For each virtual token, predict the embedding
		self.config.n_virtual_tokens = self.args.total_virtual_tokens
		self.config.word_embedding_dim = self.args.word_embedding_dim
		
		# Set params
		dropout_prob = config.hidden_dropout_prob if hasattr(config,
															 'hidden_dropout_prob') else config.dropout_rate if hasattr(
			config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config,
																										   'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		if MLP_h is None:
			MLP_h = hidden_dim
		
		# Define the head for encoding the row vectors - weighs virtual tokens
		self.row_dropout = torch.nn.Dropout(dropout_prob)
		self.row_down_proj = torch.nn.Linear(hidden_dim, MLP_h)
		self.row_up_proj = torch.nn.Linear(MLP_h, self.config.n_virtual_tokens * self.rank, bias=use_bias)
		
		# Define the head for encoding the column vectors - weighs the word embedding dimensions
		self.col_dropout = torch.nn.Dropout(dropout_prob)
		self.col_down_proj = torch.nn.Linear(hidden_dim, MLP_h)
		self.col_up_proj = torch.nn.Linear(MLP_h, self.config.word_embedding_dim * self.rank, bias=use_bias)
		
		self.init_predictor_head()
	
	def init_predictor_head(self):
		# Initialize the weights for the row head
		self.row_down_proj.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.row_up_proj.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.row_up_proj.bias.data.zero_()
		
		# Initialize the weights for the column head
		self.col_down_proj.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.col_up_proj.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.col_up_proj.bias.data.zero_()
	
	def get_instance_embedding(self, input_ids, attention_mask=None, token_type_ids=None):
		
		if attention_mask is None:
			# Attend to all tokens
			attention_mask = torch.ones_like(input_ids)
			attention_mask = attention_mask.to(device=input_ids.device)
		
		# Get the CLS token embedding
		if self.args.enc_model_type == 'roberta-large':
			x = self.base(
				input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)
			x = x[0]
			x = x[:, 0, :]  # take <s> which is the first token as seq. representation (equiv. to [CLS])
		else:
			raise NotImplementedError
		return x
	
	def forward(self, input_ids, attention_mask=None, token_type_ids=None):
		
		if self.freeze_base:
			with torch.no_grad():
				# Get the instance embedding
				inst_embedding = self.get_instance_embedding(input_ids, attention_mask, token_type_ids)
				inst_embedding = inst_embedding.detach()
		
		else:
			# Get the instance embedding
			inst_embedding = self.get_instance_embedding(input_ids, attention_mask, token_type_ids)
		
		# Predict the row weights
		row_weights = self.row_dropout(inst_embedding)
		row_weights = self.row_down_proj(row_weights)
		row_weights = torch.nn.functional.tanh(row_weights)
		row_weights = self.row_up_proj(row_weights)
		row_weights = row_weights.view(-1, self.config.n_virtual_tokens, self.rank)
		
		# Predict the column weights
		col_weights = self.col_dropout(inst_embedding)
		col_weights = self.col_down_proj(col_weights)
		col_weights = torch.nn.functional.tanh(col_weights)
		col_weights = self.col_up_proj(col_weights)
		col_weights = col_weights.view(-1, self.config.word_embedding_dim, self.rank)
		
		# [Older] Multiply: uk ∈ R^l, vk ∈ R^d -> uk * vk^T ∈ R^(l x d)
		# prompt_specific_clf_embedding = torch.einsum('bi,bj->bij', row_weights, col_weights)
		
		# [Latest] Multiply: uk ∈ R^l x r, vk ∈ R^d x r -> uk * vk^T ∈ R^(l x d)
		prompt_specific_clf_embedding = torch.einsum('bir,bjr->bij', row_weights, col_weights)
		
		return prompt_specific_clf_embedding
	
	def __str__(self):
		return f"MyEncoder/{self.args.enc_model_type}"
	
	def __repr__(self):
		return f"MyEncoder/{self.args.enc_model_type}"


class IDPGSoftPromptGenerator(torch.nn.Module):
	def __init__(self, args, use_bias=True, MLP_h=256):
		super(IDPGSoftPromptGenerator, self).__init__()
		
		config, base = load_base_model(
			args,
			model_type=args.enc_model_type,
			model_name_or_path=get_huggingface_path(args.enc_model_type)
		)
		
		self.args = args
		self.config = config
		self.base = base
		
		# Base model does not require any training - freeze the weights
		for param in self.base.parameters():
			param.requires_grad = False
		
		# For each virtual token, predict the embedding
		self.config.n_virtual_tokens = self.args.total_virtual_tokens
		self.config.word_embedding_dim = self.args.word_embedding_dim
		
		# Set params [Should be same as the model used for the base]
		dropout_prob = config.hidden_dropout_prob if hasattr(config,
															 'hidden_dropout_prob') else config.dropout_rate if hasattr(
			config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config,
																										   'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		if MLP_h is None:
			MLP_h = hidden_dim
		
		# Define the head for encoding all virtual tokens
		self._dropout = torch.nn.Dropout(dropout_prob)
		self.layer_down_project = torch.nn.Linear(hidden_dim, MLP_h, bias=use_bias)
		self.layer_up_project = torch.nn.Linear(MLP_h, self.config.n_virtual_tokens * self.config.word_embedding_dim,
												bias=use_bias)
		
		self.init_predictor_head()
	
	def init_predictor_head(self):
		# Initialize the weights for the row head
		self.layer_down_project.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.layer_up_project.weight.data.normal_(mean=0.0, std=self.config_initializer_range)
		self.layer_up_project.bias.data.zero_()
	
	@torch.no_grad()
	def get_instance_embedding(self, input_ids, attention_mask=None, token_type_ids=None):
		if attention_mask is None:
			# Attend to all tokens
			attention_mask = torch.ones_like(input_ids)
			attention_mask = attention_mask.to(device=input_ids.device)
		
		# Get the CLS token embedding
		if self.args.enc_model_type == 'roberta-large':
			x = self.base(
				input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)
			x = x[0]
			x = x[:, 0, :]  # take <s> which is the first token as seq. representation (equiv. to [CLS])
		else:
			raise NotImplementedError
		return x.detach()
	
	def forward(self, input_ids, attention_mask=None, token_type_ids=None):
		
		inst_embedding = self.get_instance_embedding(input_ids, attention_mask, token_type_ids)
		
		# Predict the row weights
		soft_prompt_embedding = self._dropout(inst_embedding)
		soft_prompt_embedding = self.layer_down_project(soft_prompt_embedding)
		soft_prompt_embedding = torch.nn.functional.tanh(soft_prompt_embedding)
		soft_prompt_embedding = self.layer_up_project(soft_prompt_embedding)
		
		# Reshape [B, N * D] -> [B, N, D]
		soft_prompt_embedding = soft_prompt_embedding.view(-1, self.config.n_virtual_tokens,
														   self.config.word_embedding_dim)
		return soft_prompt_embedding
	
	def __str__(self):
		return f"IDPG/{self.args.enc_model_type}"
	
	def __repr__(self):
		return f"IDPG/{self.args.enc_model_type}"


class PHMLayer(nn.Module):
	
	def __init__(self, n, in_features, out_features):
		super(PHMLayer, self).__init__()
		
		assert out_features % n == 0, "out_features should be divisible by n"
		assert in_features % n == 0, "in_features should be divisible by n"
		
		self.n = n
		self.in_features = in_features
		self.out_features = out_features
		
		self.bias = Parameter(torch.Tensor(out_features))
		
		self.a = Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
		
		self.s = Parameter(
			torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features // n, self.in_features // n))))
		
		self.weight = torch.zeros((self.out_features, self.in_features))
		
		fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
		bound = 1 / math.sqrt(fan_in)
		init.uniform_(self.bias, -bound, bound)
	
	def kronecker_product1(self, a, b):  # adapted from Bayer Research's implementation
		siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
		res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
		siz0 = res.shape[:-4]
		out = res.reshape(siz0 + siz1)
		return out
	
	def forward(self, input: Tensor) -> Tensor:
		self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
		input = input.type(dtype=self.weight.type())
		return F.linear(input, weight=self.weight, bias=self.bias)
	
	def extra_repr(self) -> str:
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None)
	
	def reset_parameters(self) -> None:
		init.kaiming_uniform_(self.a, a=math.sqrt(5))
		init.kaiming_uniform_(self.s, a=math.sqrt(5))
		fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
		bound = 1 / math.sqrt(fan_in)
		init.uniform_(self.bias, -bound, bound)


class IDPGSoftPromptGenerator_wPHM(torch.nn.Module):
	def __init__(self, args, use_bias=True, MLP_h=256, n=16):
		super(IDPGSoftPromptGenerator_wPHM, self).__init__()
		
		config, base = load_base_model(
			args,
			model_type=args.enc_model_type,
			model_name_or_path=get_huggingface_path(args.enc_model_type)
		)
		
		self.args = args
		self.config = config
		self.base = base
		
		# Base model does not require any training - freeze the weights
		for param in self.base.parameters():
			param.requires_grad = False
		
		# For each virtual token, predict the embedding
		self.config.n_virtual_tokens = self.args.total_virtual_tokens
		self.config.word_embedding_dim = self.args.word_embedding_dim
		
		# Set params [Should be same as the model used for the base]
		dropout_prob = config.hidden_dropout_prob if hasattr(config,
															 'hidden_dropout_prob') else config.dropout_rate if hasattr(
			config, 'dropout_rate') else 0.1
		hidden_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.embed_dim if hasattr(config,
																										   'embed_dim') else 768
		self.config_initializer_range = config.initializer_range if hasattr(config, 'initializer_range') else 0.02
		
		if MLP_h is None:
			MLP_h = hidden_dim
		
		# Define the head for encoding all virtual tokens
		self._dropout = torch.nn.Dropout(dropout_prob)
		self.layer_down_project = PHMLayer(in_features=hidden_dim, out_features=MLP_h, n=n)
		self.layer_up_project = PHMLayer(in_features=MLP_h, out_features=self.config.n_virtual_tokens * self.config.word_embedding_dim, n=n)
	
	@torch.no_grad()
	def get_instance_embedding(self, input_ids, attention_mask=None, token_type_ids=None):
		if attention_mask is None:
			# Attend to all tokens
			attention_mask = torch.ones_like(input_ids)
			attention_mask = attention_mask.to(device=input_ids.device)
		
		# Get the CLS token embedding
		if self.args.enc_model_type == 'roberta-large':
			x = self.base(
				input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids
			)
			x = x[0]
			x = x[:, 0, :]  # take <s> which is the first token as seq. representation (equiv. to [CLS])
		else:
			raise NotImplementedError
		return x.detach()
	
	def forward(self, input_ids, attention_mask=None, token_type_ids=None):
		
		inst_embedding = self.get_instance_embedding(input_ids, attention_mask, token_type_ids)
		
		# Predict the row weights
		soft_prompt_embedding = self._dropout(inst_embedding)
		soft_prompt_embedding = self.layer_down_project(soft_prompt_embedding)
		soft_prompt_embedding = torch.nn.functional.tanh(soft_prompt_embedding)
		soft_prompt_embedding = self.layer_up_project(soft_prompt_embedding)
		
		# Reshape [B, N * D] -> [B, N, D]
		soft_prompt_embedding = soft_prompt_embedding.view(-1, self.config.n_virtual_tokens,
														   self.config.word_embedding_dim)
		return soft_prompt_embedding
		
	
	def __str__(self):
		return f"IDPG/{self.args.enc_model_type}"
	
	def __repr__(self):
		return f"IDPG/{self.args.enc_model_type}"


class ClassificationHead(nn.Module):
	"""Head for sentence-level classification tasks."""
	
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		classifier_dropout = (
			config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
		)
		self.dropout = nn.Dropout(classifier_dropout)
		self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
		self.pos_cls_token: int = 0 + self.config.total_virtual_tokens
	
	def forward(self, features, **kwargs):
		x = features[:, self.pos_cls_token, :]  # take <s> token (equiv. to [CLS])
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.tanh(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x


class ModelForSequenceClassification(nn.Module):
	"""Base class for all models that do sequence classification."""
	
	def __init__(self, config, base):
		super().__init__()
		self.config = config
		self.base = base
		self.num_labels = config.num_labels
		self.classifier = ClassificationHead(config)
	
	def forward(
			self,
			latent_prompt,
			input_ids,
			attention_mask=None,
			labels=None,
			return_dict: Optional[bool] = True,
	):
		outputs = self.base(latent_prompt=latent_prompt, input_ids=input_ids, attention_mask=attention_mask)
		sequence_output = outputs[0]
		logits = self.classifier(sequence_output)
		
		loss = None
		if labels is not None:
			# move labels to correct device to enable model parallelism
			labels = labels.to(logits.device)
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"
			
			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(logits, labels)
		
		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output
		
		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
	
	def get_nb_trainable_parameters(self):
		# Get the number of trainable and all parameters for sequence embedder
		seq_emb_trainable_params, seq_emb_all_params = self.base.get_nb_trainable_parameters()
		# Get the number of trainable parameters for classifier
		classifier_trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
		# Get the number of all parameters for classifier
		classifier_all_params = sum(p.numel() for p in self.classifier.parameters())
		return seq_emb_trainable_params + classifier_trainable_params, seq_emb_all_params + classifier_all_params


class LOPA(torch.nn.Module):
	
	def __init__(
			self,
			config,
			latent_prompt_gen: LatentPromptAttentionGenerator,
			seq_classifier: PeftLopaModelForMaskedLM
	):
		super(LOPA, self).__init__()
		self.latent_prompt_gen = latent_prompt_gen
		self.seq_classifier = seq_classifier
		
		# Set the config same as the classifier encoder
		self.config = config
	
	def forward(self, batch):
		# Encode the attention weights for the latent prompt
		att_logits = self.latent_prompt_gen(
			input_ids=batch['enc_input_ids'],
			attention_mask=batch['enc_attention_mask'],
			token_type_ids=batch['enc_token_type_ids']
		)
		latent_att_weights = torch.sigmoid(att_logits)
		
		# Shift the position of the mask tokens to the right by total_virtual_tokens
		batch['mask_pos'] = batch['mask_pos'] + self.config.total_virtual_tokens
		
		# Call the sequence classifier
		output = self.seq_classifier(
			latent_prompt_att_weights=latent_att_weights,
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
			mask_pos=batch['mask_pos'],
			labels=batch['labels']
		)
		
		return output


class IDPG(torch.nn.Module):
	
	def __init__(
			self,
			config,
			latent_prompt_gen: IDPGSoftPromptGenerator,
			seq_classifier: PeftIDPGModelForMaskedLM
	):
		super(IDPG, self).__init__()
		self.latent_prompt_gen = latent_prompt_gen
		self.seq_classifier = seq_classifier
		
		# Set the config same as the classifier
		self.config = config
	
	def forward(self, batch):
		# Generate the soft (latent) prompt
		soft_prompt = self.latent_prompt_gen(
			input_ids=batch['enc_input_ids'],
			attention_mask=batch['enc_attention_mask'],
			token_type_ids=batch['enc_token_type_ids']
		)
		
		# Shift the position of the mask tokens to the right by total_virtual_tokens
		batch['mask_pos'] = batch['mask_pos'] + self.config.total_virtual_tokens
		
		# Call the sequence classifier
		output = self.seq_classifier(
			soft_prompt=soft_prompt,
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
			mask_pos=batch['mask_pos'],
			labels=batch['labels']
		)
		
		return output


class DummyModel(torch.nn.Module):
	def __init__(self, config, seq_classifier: PeftModelForMaskedLM):
		super(DummyModel, self).__init__()
		self.seq_classifier = seq_classifier
		
		# Set the config same as the classifier
		self.config = config
	
	def forward(self, batch):
		# Shift the position of the mask tokens to the right by total_virtual_tokens
		batch['mask_pos'] = batch['mask_pos'] + self.config.total_virtual_tokens
		
		# Call the sequence classifier
		output = self.seq_classifier(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
			mask_pos=batch['mask_pos'],
			labels=batch['labels']
		)
		
		return output
