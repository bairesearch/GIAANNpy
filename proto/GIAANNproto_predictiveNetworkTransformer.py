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


"""

import torch
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from GIAANNproto_globalDefs import *

	

def nextWordPredictionTransformerCreate(databaseNetworkObject):
	global model, criterion, optimizer, batch_size

	num_layers = 3  # Number of transformer layers

	# Instantiate the model
	#NextWordPredictionTransformerModel
	model = CustomTransformer(databaseNetworkObject.p, databaseNetworkObject.s, databaseNetworkObject.c, databaseNetworkObject.f, num_layers)
	model.train()  # set model to training mode

	if(multipleTargets):
		criterion = nn.BCEWithLogitsLoss()
	else:
		criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0005)
	batch_size = 1
	
def nextWordPredictionTransformerTrainStep(global_feature_neurons, database_feature_connections, targets):
	global model, criterion, optimizer, batch_size
	
	global_feature_neurons = global_feature_neurons.unsqueeze(0)	#add batch dim (not used)
	targets = targets.unsqueeze(0)	#add batch dim

	global_feature_neurons = global_feature_neurons.to_dense()
	if(useGPUdense and not useGPUsparse):
		global_feature_neurons = global_feature_neurons.to(deviceDense)
		
	outputs = model(global_feature_neurons, database_feature_connections)  # Outputs shape: (batch_size, c, f)

	loss = criterion(outputs, targets)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	loss_value = loss.item()
	print("loss_value = ", loss_value)

	return getTopkPredictions(outputs)
	
def getTopkPredictions(outputs):

	with pt.no_grad():

		if(multipleTargets):
			# Apply sigmoid to get probabilities
			probs = pt.sigmoid(outputs)  # Shape: (batch_size, c, f)
		else:
			probs = outputs

		column_probs = probs.mean(dim=2)  # Shape: (batch_size, c)
		_, concept_columns_indices_next = pt.topk(column_probs, kc, dim=1)

		# For each of the top kc columns, compute top kf features
		top_kf_indices = []
		for column_idx in concept_columns_indices_next:
			column_data = probs[:, column_idx, :]  # Shape: (batch_size, f)
			feature_probs = column_data.mean(dim=0)  # Shape: (f,)
			topk_feature_probs, topk_feature_indices = pt.topk(feature_probs, kf)  # Shapes: (kf,), (kf,)
			top_kf_indices.append(topk_feature_indices)

		concept_columns_feature_indices_next = pt.stack(top_kf_indices)  # Shape: (batch_size, kc, kf)

	concept_columns_indices_next = concept_columns_indices_next[0]	#select first sample of batch
	concept_columns_feature_indices_next = concept_columns_feature_indices_next[0]	#select first sample of batch

	return concept_columns_indices_next, concept_columns_feature_indices_next


class InputAttentionLayer(nn.Module):
	def __init__(self, p, s, c, f):
		super(InputAttentionLayer, self).__init__()
		self.p = p  # Number of properties per feature
		self.s = s  # Number of input segments
		self.c = c  # Number of columns (sequence length)
		self.f = f  # Number of input features

		# Linear layer for V (values), applied over the feature dimension
		self.W_v = nn.Linear(f, f, bias=False)

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
				attention_output = torch.bmm(attention_weights, V_head)  # Shape: (batch_size, c, f)

				attention_outputs.append(attention_output)

		# Concatenate attention outputs from all heads along the embedding dimension
		# attention_outputs is a list of tensors with shape (batch_size, c, f)
		attention_outputs = torch.cat(attention_outputs, dim=2)  # Shape: (batch_size, c, p*s*f)

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
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.p_hidden, dim_feedforward=self.f_mlp)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# MLP for deriving the predicted token
		self.fc = nn.Linear(self.embedding_dim, c * f)

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
		final_output = transformer_output[-1]  # Shape: (batch_size, embedding_dim)

		# Apply the MLP to derive the logits
		logits = self.fc(final_output)  # Shape: (batch_size, c * f)

		# Reshape logits to (batch_size, c, f)
		logits = logits.view(batch_size, self.c, self.f)

		if(multipleTargets):
			probabilities = logits
		else:
			# Apply softmax to get probabilities over (c, f)
			logits_flat = logits.view(batch_size, -1)  # Shape: (batch_size, c * f)
			probabilities_flat = F.softmax(logits_flat, dim=-1)
			probabilities = probabilities_flat.view(batch_size, self.c, self.f)  # Shape: (batch_size, c, f)

		return probabilities  # Shape: (batch_size, c, f)


