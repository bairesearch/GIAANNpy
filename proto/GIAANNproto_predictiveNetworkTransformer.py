"""GIAANNproto_predictiveNetworkTransformer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto predictive Network Transformer

FUTURE: consider pytorch implementation for sparse tensors.

"""

import torch
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from GIAANNproto_globalDefs import *
import GIAANNproto_predictiveNetworkOperations
	

def nextWordPredictionTransformerCreate(databaseNetworkObject):
	global model, criterion, criterion_c, criterion_f, optimizer, batch_size

	num_layers = 3  # Number of transformer layers
	
	print("nextWordPredictionTransformerCreate:")
	print("databaseNetworkObject.c = ", databaseNetworkObject.c)
	print("databaseNetworkObject.f = ", databaseNetworkObject.f)
	
	# Instantiate the model
	#NextWordPredictionTransformerModel
	model = CustomTransformer(databaseNetworkObject.p, databaseNetworkObject.s, databaseNetworkObject.c, databaseNetworkObject.f, num_layers)
	model = model.to(devicePredictiveNetworkModel)
	model.train()  # set model to training mode

	if(inferencePredictiveNetworkIndependentFCpredictions):
		if multipleTargets:
			criterion_c = nn.BCEWithLogitsLoss()
			criterion_f = nn.BCEWithLogitsLoss()
		else:
			criterion_c = nn.CrossEntropyLoss()
			criterion_f = nn.CrossEntropyLoss()
	else:
		if(multipleTargets):
			criterion = nn.BCEWithLogitsLoss()
		else:
			criterion = nn.MSELoss()
	
	optimizer = optim.Adam(model.parameters(), lr=inferencePredictiveNetworkLearningRate)
	batch_size = 1
	
def nextWordPredictionTransformerTrainStep(global_feature_neurons, database_feature_connections, targets, targets_c, targets_f):
	global model, criterion, criterion_c, criterion_f, optimizer, batch_size
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		targets_c = targets_c.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
		targets_f = targets_f.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	else:
		targets = targets.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	
	global_feature_neurons = global_feature_neurons.unsqueeze(0)	#add batch dim (not used)
	global_feature_neurons = global_feature_neurons.to(devicePredictiveNetworkModel)
	global_feature_neurons = global_feature_neurons.to_dense()
	
	#print("global_feature_neurons.shape = ", global_feature_neurons.shape)
	#print("global_feature_neurons.shape = ", global_feature_neurons.shape)
		

	if(inferencePredictiveNetworkIndependentFCpredictions):
		outputs_c, outputs_f = model(global_feature_neurons, database_feature_connections)  # Outputs shape: (batch_size, c, f)
		loss_c = criterion_c(outputs_c, targets_c)
		loss_f = criterion_f(outputs_f, targets_f)
		loss = loss_c + loss_f  # Combine losses
	else:
		outputs = model(global_feature_neurons, database_feature_connections)  # Outputs shape: (batch_size, c, f)
		loss = criterion(outputs, targets)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	loss_value = loss.item()
	print("loss_value = ", loss_value)
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		topk_c = GIAANNproto_predictiveNetworkOperations.getTopkPredictionsC(outputs_c)
		topk_f = GIAANNproto_predictiveNetworkOperations.getTopkPredictionsF(outputs_f)
		return topk_c, topk_f
	else:
		topk = GIAANNproto_predictiveNetworkOperations.getTopkPredictions(outputs)
		return topk
	
	
class InputAttentionLayer(nn.Module):
	def __init__(self, p, s, c, f):
		super(InputAttentionLayer, self).__init__()
		self.p = p  # Number of properties per feature
		self.s = s  # Number of input segments
		self.c = c  # Number of columns (sequence length)
		self.f = f  # Number of input features

		# Linear layer for V (values), applied over the feature dimension
		self.W_v = nn.Linear(f, f, bias=False, device=devicePredictiveNetworkModel)

	def forward(self, X, database_feature_connections):
		# X shape: (batch_size, p, s, c, f)
		# database_feature_connections shape: (p, s, c, f, c, f)

		batch_size = X.size(0)

		# Reshape X to apply W_v over the last dimension (f)
		X_reshaped = X.view(batch_size, self.p * self.s * self.c, self.f)
		V = self.W_v(X_reshaped)  # Shape: (batch_size, p*s*c, f)

		# Reshape V back to (batch_size, p, s, c, f)
		V = V.view(batch_size, self.p, self.s, self.c, self.f)

		attention_outputs = []

		# Iterate over each attention head (total of p * s heads)
		for p_idx in range(self.p):
			for s_idx in range(self.s):
				# Extract V for the current head
				V_head = V[:, p_idx, s_idx, :, :]  # Shape: (batch_size, c, f)

				# Extract the corresponding attention scores from database_feature_connections
				# KQ_head shape: (c, f, c, f)
				KQ_head = database_feature_connections[p_idx, s_idx]  # Shape: (c, f, c, f)

				# Sum over the feature dimensions to obtain attention weights between columns
				# First, sum over source feature dimension (from features)
				KQ_summed_over_from_f = KQ_head.sum(dim=1)  # Shape: (c, c, f)
				# Then, sum over target feature dimension (to features)
				attention_scores = KQ_summed_over_from_f.sum(dim=2)  # Shape: (c, c)

				attention_scores = attention_scores.to_dense()
				if(useGPUdense and not useGPUsparse):
					attention_scores = attention_scores.to(deviceDense)
					
				# Apply softmax to get attention weights
				attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (c, c)

				# Expand attention_weights to match batch size
				attention_weights = attention_weights.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, c, c)

				# Apply attention weights to V_head
				attention_output = pt.bmm(attention_weights, V_head)  # Shape: (batch_size, c, f)

				attention_outputs.append(attention_output)

		# Concatenate attention outputs from all heads along the embedding dimension
		# attention_outputs is a list of tensors with shape (batch_size, c, f)
		attention_outputs = pt.cat(attention_outputs, dim=2)  # Shape: (batch_size, c, p*s*f)

		# Permute to match transformer input requirements
		# Transformer expects input of shape (sequence_length, batch_size, embedding_dim)
		attention_outputs = attention_outputs.permute(1, 0, 2).contiguous()  # Shape: (c, batch_size, p*s*f)

		return attention_outputs  # Shape: (sequence_length, batch_size, embedding_dim)


class CustomTransformer(nn.Module):
	def __init__(self, p, s, c, f, num_layers):
		super(CustomTransformer, self).__init__()
		self.p = p
		self.s = s
		self.c = c
		self.f = f
		self.p_hidden = p*s
		self.num_layers = num_layers
		self.embedding_dim = self.p * self.s * self.f	#embedding dimension includes concatenation of all heads
		self.f_mlp = self.embedding_dim*4
		
		if(transformerUseInputConnections):
			# Custom attention layer for the input layer
			self.input_attention_layer = InputAttentionLayer(p, s, c, f)

		# Transformer encoder layers for the hidden layers
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.p_hidden, dim_feedforward=self.f_mlp, device=devicePredictiveNetworkModel)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		if(inferencePredictiveNetworkInitialiseWeightsNearZero):
			#self.initialise_encoder_weights_zero()	#debug only
			self.initialise_encoder_weights_near_zero()

		# MLP for deriving the predicted token
		if(transformerOutputLayerUseEveryColumn):
			final_dim = c * self.embedding_dim
		else:
			final_dim = self.embedding_dim
			
		if(inferencePredictiveNetworkIndependentFCpredictions):
			self.fc_c = nn.Linear(final_dim, self.c, device=devicePredictiveNetworkModel)
			self.fc_f = nn.Linear(final_dim, self.f, device=devicePredictiveNetworkModel)
		else:
			self.fc = nn.Linear(final_dim, c * f, device=devicePredictiveNetworkModel)
		
	def forward(self, X, database_feature_connections):
		# X shape: (batch_size, p, s, c, f)
		# database_feature_connections shape: (p, s, c, f, c, f)
		batch_size = X.size(0)

		embedding = X.permute(3, 0, 1, 2, 4).contiguous()  # Shape: (c, batch_size, p, s, f)
		embedding = embedding.view(self.c, batch_size, self.p * self.s * self.f)
			
		if(transformerUseInputConnections):
			# Apply the custom attention layer
			input_attention_output = self.input_attention_layer(X, database_feature_connections)
			# attention_outputs shape: (sequence_length, batch_size, embedding_dim)
			embedding = (embedding + input_attention_output) / 2.0	#residual connection from X
		
		# Pass through the transformer encoder
		transformer_output = self.transformer_encoder(embedding)
		# transformer_output shape: (sequence_length, batch_size, embedding_dim)

		# Use the output from the last sequence position
		if(transformerOutputLayerUseEveryColumn):
			final_output = transformer_output.view(batch_size, self.c * self.embedding_dim)	# Shape: (batch_size, c*embedding_dim)
		else:
			final_output = transformer_output[-1]  # Shape: (batch_size, embedding_dim)

		if(inferencePredictiveNetworkIndependentFCpredictions):
			# Apply the separate MLPs to derive logits for c and f
			logits_c = self.fc_c(final_output)  # Shape: (batch_size, c)
			logits_f = self.fc_f(final_output)  # Shape: (batch_size, f)
			return logits_c, logits_f  # Two separate outputs	 # Shape: (batch_size, c), Shape: (batch_size, f)
			'''
			if multipleTargets:
				probabilities_c = logits_c
				probabilities_f = logits_f
			else:
				probabilities_c = F.softmax(logits_c, dim=-1)  # Shape: (batch_size, c)
				probabilities_f = F.softmax(logits_f, dim=-1)  # Shape: (batch_size, f)
			return probabilities_c, probabilities_f  # Two separate outputs	 # Shape: (batch_size, c), Shape: (batch_size, f)
			'''
		else:
			# Apply the MLP to derive the logits
			logits = self.fc(final_output)  # Shape: (batch_size, c * f)
			logits = logits.view(batch_size, self.c, self.f)
			return logits  # Shape: (batch_size, c, f)
			'''
			if(multipleTargets):
				probabilities = logits
			else:
				#Apply softmax to get probabilities over (c, f)
				logits_flat = logits.view(batch_size, -1)  # Shape: (batch_size, c * f)
				probabilities_flat = F.softmax(logits_flat, dim=-1)
				probabilities = probabilities_flat.view(batch_size, self.c, self.f)  # Shape: (batch_size, c, f)
			return probabilities  # Shape: (batch_size, c, f)
			'''
		
	def initialise_encoder_weights_near_zero(self, init_std = 1e-5):
		# 3. Initialize each layer to be near-identity.
		for layer in self.transformer_encoder.layers:
			# --- Multi-head attention ---
			# Small random initialization instead of exact zeros
			nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=init_std)
			nn.init.zeros_(layer.self_attn.in_proj_bias)
			nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=init_std)
			nn.init.zeros_(layer.self_attn.out_proj.bias)

			# --- Feedforward sub-layer ---
			nn.init.normal_(layer.linear1.weight, mean=0.0, std=init_std)
			nn.init.zeros_(layer.linear1.bias)
			nn.init.normal_(layer.linear2.weight, mean=0.0, std=init_std)
			nn.init.zeros_(layer.linear2.bias)

			# --- LayerNorm parameters ---
			# Keep scale=1.0 and bias=0.0 so LN doesn't shift/scale significantly
			nn.init.ones_(layer.norm1.weight)
			nn.init.zeros_(layer.norm1.bias)
			nn.init.ones_(layer.norm2.weight)
			nn.init.zeros_(layer.norm2.bias)

	def initialise_encoder_weights_zero(self):
		# 3. Initialize each layer to be near-identity.
		for layer in self.transformer_encoder.layers:
			# --- Multi-head attention ---
			nn.init.zeros_(layer.self_attn.in_proj_weight)   # Q/K/V weights
			nn.init.zeros_(layer.self_attn.in_proj_bias)     
			nn.init.zeros_(layer.self_attn.out_proj.weight)  # Output projection
			nn.init.zeros_(layer.self_attn.out_proj.bias)

			# --- Feedforward sub-layer ---
			nn.init.zeros_(layer.linear1.weight)
			nn.init.zeros_(layer.linear1.bias)
			nn.init.zeros_(layer.linear2.weight)
			nn.init.zeros_(layer.linear2.bias)

			# --- LayerNorm parameters ---
			# Keep default LN scale=1.0 and bias=0.0 for minimal shift/scale
			nn.init.ones_(layer.norm1.weight)
			nn.init.zeros_(layer.norm1.bias)
			nn.init.ones_(layer.norm2.weight)

			
def save_model(path, filename="model.pt"):
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    # Construct full filepath
    filepath = os.path.join(path, filename)
    
    # Save the model state_dict
    pt.save(model.state_dict(), filepath)
    return filepath


def load_model(filepath, map_location=None):
    # Load state_dict from file
    state_dict = pt.load(filepath, map_location=map_location)
    
    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    
    return model

