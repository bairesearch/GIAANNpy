"""GIAANNproto_predictionNetworkModelMLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto predictive Network MLP

FUTURE: consider pytorch implementation for sparse tensors.

"""

import torch as pt
import torch.nn as nn
import torch.optim as optim


from GIAANNproto_globalDefs import *
import GIAANNproto_predictionNetworkOperations

def nextWordPredictionModelCreate(databaseNetworkObject):
	global model, criterion, criterionC, criterionF, optimizer, batchSize

	model = NextWordPredictionMLPmodel(databaseNetworkObject)
	model = model.to(devicePredictiveNetworkModel)
	model.train()  # set model to training mode

	if(inferencePredictiveNetworkIndependentFCpredictions):
		if(multipleTargets):
			criterionC = nn.BCEWithLogitsLoss()
			criterionF = nn.BCEWithLogitsLoss()
		else:
			criterionC = nn.MSELoss()
			criterionF = nn.MSELoss()
	else:
		if(multipleTargets):
			criterion = nn.BCEWithLogitsLoss()
		else:
			criterion = nn.MSELoss()
			
	optimizer = optim.Adam(model.parameters(), lr=inferencePredictiveNetworkLearningRate)
	batchSize = 1

def nextWordPredictionMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF):
	global model, criterion, optimizer, batchSize
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		targetsC = targetsC.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
		targetsF = targetsF.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	else:
		targets = targets.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim

	globalFeatureNeurons = globalFeatureNeurons.to(devicePredictiveNetworkModel)
	globalFeatureNeurons = globalFeatureNeurons.to_dense()	#shape: (p, s, c, f) or (s, c, f)

	if(inferencePredictiveNetworkNormaliseInputs):	#and useGPUpredictiveNetworkModel
		globalFeatureNeurons = GIAANNproto_predictionNetworkOperations.normaliseDenseTensor(globalFeatureNeurons, dim=inferencePredictiveNetworkNormaliseDim)
		
	if(inferencePredictiveNetworkUseInputAllProperties):
		globalFeatureNeurons = globalFeatureNeurons.reshape(globalFeatureNeurons.shape[2], globalFeatureNeurons.shape[0]*globalFeatureNeurons.shape[1]*globalFeatureNeurons.shape[3])
	else:
		globalFeatureNeurons = globalFeatureNeurons.reshape(globalFeatureNeurons.shape[1], globalFeatureNeurons.shape[0]*globalFeatureNeurons.shape[2])
	globalFeatureNeurons = globalFeatureNeurons.unsqueeze(0)	#shape: (batchSize, c, inputSize)

	if(inferencePredictiveNetworkIndependentFCpredictions):
		outputsC, outputsF = model(globalFeatureNeurons) # Outputs shape: (batchSize, c), (batchSize, f)
		lossC = criterionC(outputsC, targetsC)
		lossF = criterionF(outputsF, targetsF)
		loss = lossC + lossF  # Combine losses
	else:	
		outputs = model(globalFeatureNeurons)  # Outputs shape: (batchSize, c, f)
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
		
class NextWordPredictionMLPmodel(nn.Module):
	def __init__(self, databaseNetworkObject):
		super(NextWordPredictionMLPmodel, self).__init__()
		self.s = databaseNetworkObject.s
		self.c = databaseNetworkObject.c
		self.f = databaseNetworkObject.f
		self.p = databaseNetworkObject.p
		
		if(inferencePredictiveNetworkUseInputAllProperties):
			inputSize = databaseNetworkObject.p * databaseNetworkObject.s * databaseNetworkObject.c * databaseNetworkObject.f
			hiddenSizeMultiplier = 1	#default: 1	#TODO: requires testing
		else:
			inputSize = databaseNetworkObject.s * databaseNetworkObject.c * databaseNetworkObject.f
			hiddenSizeMultiplier = 2	#default: 2	#TODO: requires testing
		outputSize = databaseNetworkObject.c * databaseNetworkObject.f
		hiddenSize = inputSize * hiddenSizeMultiplier

		linearList = []
		for l in range(numberOfHiddenLayers):
			if(l == 0):
				linear = nn.Linear(inputSize, hiddenSize, device=devicePredictiveNetworkModel)
				linearList.append(linear)
			else:
				linear = nn.Linear(hiddenSize, hiddenSize, device=devicePredictiveNetworkModel)
				linearList.append(linear)
			relu = nn.ReLU()
			linearList.append(relu)
		self.linear = nn.ModuleList(linearList)
		
		if(inferencePredictiveNetworkIndependentFCpredictions):
			self.linearOutC = nn.Linear(hiddenSize, self.c, device=devicePredictiveNetworkModel)
			self.linearOutF = nn.Linear(hiddenSize, self.f, device=devicePredictiveNetworkModel)
		else:
			self.linearOut = nn.Linear(hiddenSize, outputSize, device=devicePredictiveNetworkModel)
	
	def forward(self, x):
		x = x.view(x.size(0), -1)	#Flatten the input: (batchSize, c, f) -> (batchSize, c*f)
		out = x
		for layerIndex, layer in enumerate(self.linear):
			out = layer(out)  # Shape: (kcNetwork, hiddenSize)

		if(inferencePredictiveNetworkIndependentFCpredictions):
			outC = self.linearOutC(out)  # Shape: (batchSize, c)
			outF = self.linearOutF(out)  # Shape: (batchSize, f)
			outC = outC.view(batchSize, self.c)
			outF = outF.view(batchSize, self.f)
			return outC, outF
		else:
			out = self.linearOut(out)	#Output shape: (batchSize, c*f)
			out = out.view(batchSize, self.c, self.f)
			return out



def saveModel(model, path, filename="model.pt"):
	# Ensure directory exists
	os.makedirs(path, exist_ok=True)
	
	# Construct full filepath
	filepath = os.path.join(path, filename)
	
	# Save the model state_dict
	pt.save(model.state_dict(), filepath)
	return filepath


def loadModel(model, filepath, map_location=None):
	# Load state_dict from file
	stateDict = pt.load(filepath, map_location=map_location)
	
	# Load the state_dict into the model
	model.load_state_dict(stateDict)
	
	return model
