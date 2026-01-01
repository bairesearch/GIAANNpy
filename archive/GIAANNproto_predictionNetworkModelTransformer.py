"""GIAANNproto_predictionNetworkModelTransformer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto prediction Network Transformer

FUTURE: consider pytorch implementation for sparse tensors.

"""

import torch
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from GIAANNproto_globalDefs import *
import GIAANNproto_predictionNetworkOperations
	

def nextWordPredictionModelCreate(databaseNetworkObject):
	global model, criterion, criterionC, criterionF, optimizer, batchSize

	numLayers = 1  #default: 3 # Number of transformer layers
	
	print("nextWordPredictionTransformerCreate:")
	print("databaseNetworkObject.c = ", databaseNetworkObject.c)
	print("databaseNetworkObject.f = ", databaseNetworkObject.f)
	
	# Instantiate the model
	#NextWordPredictionTransformerModel
	model = CustomTransformer(databaseNetworkObject.p, databaseNetworkObject.s, databaseNetworkObject.c, databaseNetworkObject.f, numLayers)
	model = model.to(devicePredictiveNetworkModel)
	model.train()  # set model to training mode

	if(inferencePredictiveNetworkIndependentFCpredictions):
		if multipleTargets:
			criterionC = nn.BCEWithLogitsLoss()
			criterionF = nn.BCEWithLogitsLoss()
		else:
			criterionC = nn.CrossEntropyLoss()
			criterionF = nn.CrossEntropyLoss()
	else:
		if(multipleTargets):
			criterion = nn.BCEWithLogitsLoss()
		else:
			criterion = nn.MSELoss()
	
	optimizer = optim.Adam(model.parameters(), lr=inferencePredictiveNetworkLearningRate)
	batchSize = 1
	

def nextWordPredictionTransformerTrainStep(globalFeatureNeurons, databaseFeatureConnections, targets, targetsC, targetsF):
	global model, criterion, criterionC, criterionF, optimizer, batchSize
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		targetsC = targetsC.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
		targetsF = targetsF.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	else:
		targets = targets.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	
	globalFeatureNeurons = globalFeatureNeurons.unsqueeze(0)	#add batch dim (not used)
	globalFeatureNeurons = globalFeatureNeurons.to(devicePredictiveNetworkModel)
	globalFeatureNeurons = globalFeatureNeurons.to_dense()

	if(inferencePredictiveNetworkNormaliseInputs):	#and useGPUpredictiveNetworkModel
		globalFeatureNeurons = GIAANNproto_predictionNetworkOperations.normaliseDenseTensor(globalFeatureNeurons, dim=inferencePredictiveNetworkNormaliseDim)
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		outputsC, outputsF = model(globalFeatureNeurons, databaseFeatureConnections)  # Outputs shape: (batch_size, c, f)
		lossC = criterionC(outputsC, targetsC)
		lossF = criterionF(outputsF, targetsF)
		loss = lossC + lossF  # Combine losses
	else:
		outputs = model(globalFeatureNeurons, databaseFeatureConnections)  # Outputs shape: (batch_size, c, f)
		loss = criterion(outputs, targets)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	lossValue = loss.item()
	print("loss_value = ", lossValue)
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		topkC = GIAANNproto_predictionNetworkOperations.getTopkPredictionsC(outputsC)
		topkF = GIAANNproto_predictionNetworkOperations.getTopkPredictionsF(outputsF)
		return topkC, topkF
	else:
		topk = GIAANNproto_predictionNetworkOperations.getTopkPredictions(outputs)
		return topk
	
	
class InputAttentionLayer(nn.Module):
	def __init__(self, p, s, c, f):
		super(InputAttentionLayer, self).__init__()
		self.p = p  # Number of properties per feature
		self.s = s  # Number of input segments
		self.c = c  # Number of columns (sequence length)
		self.f = f  # Number of input features

		# Linear layer for V (values), applied over the feature dimension
		self.wV = nn.Linear(f, f, bias=False, device=devicePredictiveNetworkModel)

	def forward(self, X, databaseFeatureConnections):
		# X shape: (batch_size, p, s, c, f)
		# databaseFeatureConnections shape: (p, s, c, f, c, f)

		batchSize = X.size(0)

		# Reshape X to apply wV over the last dimension (f)
		if(inferencePredictiveNetworkUseInputAllProperties):
			xReshaped = X.view(batchSize, self.p * self.s * self.c, self.f)
			v = self.wV(xReshaped)  # Shape: (batch_size, p*s*c, f)
			v = v.view(batchSize, self.p, self.s, self.c, self.f)	# Reshape v back to (batch_size, p, s, c, f)
			pMax = self.p
		else:
			xReshaped = X.view(batchSize, self.s * self.c, self.f)
			v = self.wV(xReshaped)  # Shape: (batch_size, s*c, f)
			v = v.view(batchSize, self.s, self.c, self.f)	# Reshape v back to (batch_size, p, s, c, f)
			pMax = 1
		
		attentionOutputs = []

		# Iterate over each attention head (total of p * s heads)
		for pIdx in range(pMax):
			for sIdx in range(self.s):
				# Extract v for the current head
				if(inferencePredictiveNetworkUseInputAllProperties):
					vHead = v[:, pIdx, sIdx, :, :]  # Shape: (batch_size, c, f)
					# Extract the corresponding attention scores from databaseFeatureConnections
					# kqHead shape: (c, f, c, f)
					kqHead = databaseFeatureConnections[pIdx, sIdx]  # Shape: (c, f, c, f)
				else:
					vHead = v[:, sIdx, :, :]  # Shape: (batch_size, c, f)
					# Extract the corresponding attention scores from databaseFeatureConnections
					# kqHead shape: (c, f, c, f)
					kqHead = databaseFeatureConnections[sIdx]  # Shape: (c, f, c, f)

				# Sum over the feature dimensions to obtain attention weights between columns
				# First, sum over source feature dimension (from features)
				kqSummedOverFromF = kqHead.sum(dim=1)  # Shape: (c, c, f)
				# Then, sum over target feature dimension (to features)
				attentionScores = kqSummedOverFromF.sum(dim=2)  # Shape: (c, c)

				attentionScores = attentionScores.to_dense()
				if(useGPUdense and not useGPUsparse):
					attentionScores = attentionScores.to(deviceDense)
					
				# Apply softmax to get attention weights
				attentionWeights = F.softmax(attentionScores, dim=-1)  # Shape: (c, c)

				# Expand attentionWeights to match batch size
				attentionWeights = attentionWeights.unsqueeze(0).expand(batchSize, -1, -1)  # Shape: (batch_size, c, c)

				# Apply attention weights to vHead
				attentionOutput = pt.bmm(attentionWeights, vHead)  # Shape: (batch_size, c, f)

				attentionOutputs.append(attentionOutput)

		# Concatenate attention outputs from all heads along the embedding dimension
		# attentionOutputs is a list of tensors with shape (batch_size, c, f)
		attentionOutputs = pt.cat(attentionOutputs, dim=2)  # Shape: (batch_size, c, p*s*f)

		# Permute to match transformer input requirements
		# Transformer expects input of shape (sequence_length, batch_size, embedding_dim)
		attentionOutputs = attentionOutputs.permute(1, 0, 2).contiguous()  # Shape: (c, batch_size, p*s*f)

		return attentionOutputs  # Shape: (sequence_length, batch_size, embedding_dim)


class CustomTransformer(nn.Module):
	def __init__(self, p, s, c, f, numLayers):
		super(CustomTransformer, self).__init__()
		self.p = p
		self.s = s
		self.c = c
		self.f = f
		self.numLayers = numLayers
		if(inferencePredictiveNetworkUseInputAllProperties):
			self.pHidden = p*s
			self.embeddingDim = self.p * self.s * self.f	#embedding dimension includes concatenation of all heads
		else:
			self.pHidden = s
			self.embeddingDim = self.s * self.f	#embedding dimension includes concatenation of all heads
		self.fMlp = self.embeddingDim*4
		
		if(transformerUseInputConnections):
			# Custom attention layer for the input layer
			self.inputAttentionLayer = InputAttentionLayer(p, s, c, f)

		# Transformer encoder layers for the hidden layers
		encoderLayer = nn.TransformerEncoderLayer(d_model=self.embeddingDim, nhead=self.pHidden, dim_feedforward=self.fMlp, device=devicePredictiveNetworkModel)
		self.transformerEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		if(inferencePredictiveNetworkInitialiseWeightsNearZero):
			#self.initialiseEncoderWeightsZero()	#debug only
			self.initialiseEncoderWeightsNearZero()

		# MLP for deriving the predicted token
		if(transformerOutputLayerUseEveryColumn):
			finalDim = c * self.embeddingDim
		else:
			finalDim = self.embeddingDim
			
		if(inferencePredictiveNetworkIndependentFCpredictions):
			self.fcC = nn.Linear(finalDim, self.c, device=devicePredictiveNetworkModel)
			self.fcF = nn.Linear(finalDim, self.f, device=devicePredictiveNetworkModel)
		else:
			self.fc = nn.Linear(finalDim, c * f, device=devicePredictiveNetworkModel)
		
	def forward(self, X, databaseFeatureConnections):
		# X shape: (batch_size, p, s, c, f)
		# databaseFeatureConnections shape: (p, s, c, f, c, f)
		batchSize = X.size(0)

		if(inferencePredictiveNetworkUseInputAllProperties):
			embedding = X.permute(3, 0, 1, 2, 4).contiguous()  # Shape: (c, batch_size, p, s, f)
			embedding = embedding.view(self.c, batchSize, self.p * self.s * self.f)
		else:
			embedding = X.permute(2, 0, 1, 3).contiguous()  # Shape: (c, batch_size, s, f)
			embedding = embedding.view(self.c, batchSize, self.s * self.f)
			
		if(transformerUseInputConnections):
			# Apply the custom attention layer
			inputAttentionOutput = self.inputAttentionLayer(X, databaseFeatureConnections)
			# attentionOutputs shape: (sequence_length, batch_size, embedding_dim)
			embedding = (embedding + inputAttentionOutput) / 2.0	#residual connection from X
		
		# Pass through the transformer encoder
		transformerOutput = self.transformerEncoder(embedding)
		# transformerOutput shape: (sequence_length, batch_size, embedding_dim)

		# Use the output from the last sequence position
		if(transformerOutputLayerUseEveryColumn):
			finalOutput = transformerOutput.view(batchSize, self.c * self.embeddingDim)	# Shape: (batch_size, c*embeddingDim)
		else:
			finalOutput = transformerOutput[-1]  # Shape: (batch_size, embeddingDim)

		if(inferencePredictiveNetworkIndependentFCpredictions):
			# Apply the separate MLPs to derive logits for c and f
			logitsC = self.fcC(finalOutput)  # Shape: (batch_size, c)
			logitsF = self.fcF(finalOutput)  # Shape: (batch_size, f)
			return logitsC, logitsF  # Two separate outputs
			'''
			if multipleTargets:
				probabilitiesC = logitsC
				probabilitiesF = logitsF
			else:
				probabilitiesC = F.softmax(logitsC, dim=-1)  # Shape: (batch_size, c)
				probabilitiesF = F.softmax(logitsF, dim=-1)  # Shape: (batch_size, f)
			return probabilitiesC, probabilitiesF  # Two separate outputs
			'''
		else:
			# Apply the MLP to derive the logits
			logits = self.fc(finalOutput)  # Shape: (batch_size, c * f)
			logits = logits.view(batchSize, self.c, self.f)
			return logits  # Shape: (batch_size, c, f)
			'''
			if(multipleTargets):
				probabilities = logits
			else:
				#Apply softmax to get probabilities over (c, f)
				logitsFlat = logits.view(batchSize, -1)  # Shape: (batch_size, c * f)
				probabilitiesFlat = F.softmax(logitsFlat, dim=-1)
				probabilities = probabilitiesFlat.view(batchSize, self.c, self.f)
			return probabilities  # Shape: (batch_size, c, f)
			'''
		
	def initialiseEncoderWeightsNearZero(self, initStd = 1e-5):
		# 3. Initialize each layer to be near-identity.
		for layer in self.transformerEncoder.layers:
			# --- Multi-head attention ---
			nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=initStd)
			nn.init.zeros_(layer.self_attn.in_proj_bias)
			nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=initStd)
			nn.init.zeros_(layer.self_attn.out_proj.bias)

			# --- Feedforward sub-layer ---
			nn.init.normal_(layer.linear1.weight, mean=0.0, std=initStd)
			nn.init.zeros_(layer.linear1.bias)
			nn.init.normal_(layer.linear2.weight, mean=0.0, std=initStd)
			nn.init.zeros_(layer.linear2.bias)

			# --- LayerNorm parameters ---
			nn.init.ones_(layer.norm1.weight)
			nn.init.zeros_(layer.norm1.bias)
			nn.init.ones_(layer.norm2.weight)
			nn.init.zeros_(layer.norm2.bias)

	def initialiseEncoderWeightsZero(self):
		# 3. Initialize each layer to be near-identity.
		for layer in self.transformerEncoder.layers:
			# --- Multi-head attention ---
			nn.init.zeros_(layer.self_attn.in_proj_weight)
			nn.init.zeros_(layer.self_attn.in_proj_bias)
			nn.init.zeros_(layer.self_attn.out_proj.weight)
			nn.init.zeros_(layer.self_attn.out_proj.bias)

			# --- Feedforward sub-layer ---
			nn.init.zeros_(layer.linear1.weight)
			nn.init.zeros_(layer.linear1.bias)
			nn.init.zeros_(layer.linear2.weight)
			nn.init.zeros_(layer.linear2.bias)

			# --- LayerNorm parameters ---
			nn.init.ones_(layer.norm1.weight)
			nn.init.zeros_(layer.norm1.bias)
			nn.init.ones_(layer.norm2.weight)

			
def saveModel(path, fileName="model.pt"):
	# Ensure directory exists
	os.makedirs(path, exist_ok=True)
	
	# Construct full filepath
	filePath = os.path.join(path, fileName)
	
	# Save the model state_dict
	pt.save(model.state_dict(), filePath)
	return filePath


def loadModel(filePath, mapLocation=None):
	# Load state_dict from file
	stateDict = pt.load(filePath, map_location=mapLocation)
	
	# Load the state_dict into the model
	model.load_state_dict(stateDict)
	
	return model
