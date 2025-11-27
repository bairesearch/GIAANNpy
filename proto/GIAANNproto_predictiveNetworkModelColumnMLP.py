"""GIAANNproto_predictiveNetworkModelColumnMLP.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto predictive Network Column MLP

FUTURE: consider pytorch implementation for sparse tensors.

"""

import torch as pt
import torch.nn as nn
import torch.optim as optim


from GIAANNproto_globalDefs import *
import GIAANNproto_predictiveNetworkOperations
import GIAANNproto_sparseTensors

model = None
criterion = None
criterionC = None
criterionF = None
optimizer = None
batchSize = 1

def ensureModelMatchesDatabase(databaseNetworkObject):
	"""Recreate the predictive model if the database dimensions changed."""
	global model
	if model is None:
		nextWordPredictionModelCreate(databaseNetworkObject)
		return
	if (model.p != databaseNetworkObject.p or
		model.s != databaseNetworkObject.s or
		model.c != databaseNetworkObject.c or
		model.f != databaseNetworkObject.f):
		print("Reinitialising predictive Column MLP to match database dimensions")
		nextWordPredictionModelCreate(databaseNetworkObject)

def nextWordPredictionModelCreate(databaseNetworkObject):
	global model, criterion, criterionC, criterionF, optimizer, batchSize

	model = NextWordPredictionColumnMLPmodel(databaseNetworkObject)
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

def nextWordPredictionColumnMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF):
	#globalFeatureNeurons shape: kcnetwork, f
	#conceptColumnsActivationTopkConceptsIndices shape: kcnetwork
	
	global model, criterion, optimizer, batchSize
	
	globalFeatureNeurons, conceptColumnsActivationTopkConceptsIndices = selectMostActiveColumns(globalFeatureNeurons, inferencePredictiveNetworkModelFilterColumnsK)

	if(inferencePredictiveNetworkIndependentFCpredictions):
		targetsC = targetsC.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
		targetsF = targetsF.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	else:
		targets = targets.unsqueeze(0).to(devicePredictiveNetworkModel)	#add batch dim
	
	#assume batch dimension is c dimension	#shape
	globalFeatureNeurons = globalFeatureNeurons.to(devicePredictiveNetworkModel)
	globalFeatureNeurons = globalFeatureNeurons.to_dense()	#shape: (p, s, inferencePredictiveNetworkModelFilterColumnsK, f) or (s, inferencePredictiveNetworkModelFilterColumnsK, f)
	conceptColumnsActivationTopkConceptsIndices = conceptColumnsActivationTopkConceptsIndices.to(devicePredictiveNetworkModel)
	
	if(inferencePredictiveNetworkNormaliseInputs):	#and useGPUpredictiveNetworkModel
		globalFeatureNeurons = GIAANNproto_predictiveNetworkOperations.normaliseDenseTensor(globalFeatureNeurons, dim=inferencePredictiveNetworkNormaliseDim)
		
	if(inferencePredictiveNetworkUseInputAllProperties):
		globalFeatureNeurons = globalFeatureNeurons.reshape(globalFeatureNeurons.shape[2], globalFeatureNeurons.shape[0]*globalFeatureNeurons.shape[1]*globalFeatureNeurons.shape[3])	#shape: (inferencePredictiveNetworkModelFilterColumnsK, inputSize)
	else:
		globalFeatureNeurons = globalFeatureNeurons.reshape(globalFeatureNeurons.shape[1], globalFeatureNeurons.shape[0]*globalFeatureNeurons.shape[2])	#shape: (inferencePredictiveNetworkModelFilterColumnsK, inputSize)
	
	if(inferencePredictiveNetworkIndependentFCpredictions):	
		'''
		outputsCtop, outputsF = model(globalFeatureNeurons) # shape: (1), (batchSize, f)
		outputsC = pt.zeros(model.c)  # shape: (c)
		outputsC[conceptColumnsActivationTopkConceptsIndices[outputsCtop]] = 1
		'''
		outputsCNetwork, outputsF = model(globalFeatureNeurons) # shape: (inferencePredictiveNetworkModelFilterColumnsK), (f)
		outputsC = pt.zeros(model.c, device=devicePredictiveNetworkModel)  # shape: (c)
		outputsC.scatter_(dim=0, index=conceptColumnsActivationTopkConceptsIndices, src=outputsCNetwork)

		#print("outputsC = ", outputsC)
		#print("outputsF = ", outputsF)
	
		outputsC = outputsC.unsqueeze(0)	#add batch dim
		outputsF = outputsF.unsqueeze(0)	#add batch dim
		lossC = criterionC(outputsC, targetsC)
		lossF = criterionF(outputsF, targetsF)
		loss = lossC + lossF  # Combine losses
	else:	
		outputsNetwork = model(globalFeatureNeurons)  # shape: (kcnetwork, f)
		outputs = pt.zeros((model.c, model.f), device=devicePredictiveNetworkModel)  # shape: (c, f)
		outputs.scatter_(dim=0, index=conceptColumnsActivationTopkConceptsIndices.unsqueeze(1).expand(-1, modelf.f), src=outputsNetwork)
		
		loss = criterion(outputs, targets)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	lossValue = loss.item()
	print("loss_value = ", lossValue)

	if(inferencePredictiveNetworkIndependentFCpredictions):
		topkC = GIAANNproto_predictiveNetworkOperations.getTopkPredictionsC(outputsC)
		topkF = GIAANNproto_predictiveNetworkOperations.getTopkPredictionsF(outputsF)
		return topkC, topkF
	else:
		topk = GIAANNproto_predictiveNetworkOperations.getTopkPredictions(outputs)
		return topk
	
class NextWordPredictionColumnMLPmodel(nn.Module):
	def __init__(self, databaseNetworkObject):
		super(NextWordPredictionColumnMLPmodel, self).__init__()
		self.p = databaseNetworkObject.p
		self.s = databaseNetworkObject.s
		self.c = databaseNetworkObject.c
		self.f = databaseNetworkObject.f
		
		print("databaseNetworkObject.p = ", databaseNetworkObject.p)
		print("databaseNetworkObject.s = ", databaseNetworkObject.s)
		print("databaseNetworkObject.c = ", databaseNetworkObject.c)
		print("databaseNetworkObject.f = ", databaseNetworkObject.f)
		
		if(inferencePredictiveNetworkUseInputAllProperties):
			inputSize = databaseNetworkObject.p * databaseNetworkObject.s * databaseNetworkObject.f
			hiddenSizeMultiplier = 4	#high:12	#default:4	#orig: 1	#TODO: requires testing
		else:
			inputSize = databaseNetworkObject.s * databaseNetworkObject.f
			hiddenSizeMultiplier = 8	#high:24	#default:8	#orig: 2	#TODO: requires testing
		outputSize = databaseNetworkObject.f
		hiddenSize = inputSize * hiddenSizeMultiplier
		self.hiddenSize = hiddenSize
		
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
			#self.linearOutC = nn.Linear(inferencePredictiveNetworkModelFilterColumnsK*hiddenSize, inferencePredictiveNetworkModelFilterColumnsK, device=devicePredictiveNetworkModel)
			self.linearOutC = nn.Linear(self.hiddenSize, 1, bias=True, device=devicePredictiveNetworkModel)	
			self.linearOutF = nn.Linear(hiddenSize, self.f, device=devicePredictiveNetworkModel)
		else:
			self.linearOut = nn.Linear(hiddenSize, outputSize, device=devicePredictiveNetworkModel)
		
	def forward(self, x):
		# x Shape: (inferencePredictiveNetworkModelFilterColumnsK, f)	#assume batch dimension is c dimension

		out = x
		for layerIndex, layer in enumerate(self.linear):
			out = layer(out)  # Shape: (inferencePredictiveNetworkModelFilterColumnsK, hiddenSize)
		
		if(inferencePredictiveNetworkIndependentFCpredictions):
			scoresC = self.linearOutC(out).squeeze(-1)	# shape: (inferencePredictiveNetworkModelFilterColumnsK)
			outC = pt.softmax(scoresC, dim=0)  # shape: (inferencePredictiveNetworkModelFilterColumnsK)
			
			# Weighted combination of all columns
			inF = (out * outC.unsqueeze(1)).sum(dim=0)         # shape: (hiddenSize)
			outF = self.linearOutF(inF)            # shape: (f)

			return outC, outF
		else:
			out = self.linearOut(out)	#Output shape: (inferencePredictiveNetworkModelFilterColumnsK, f)
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

def selectMostActiveColumns(globalFeatureNeurons, kc):
	if(inferencePredictiveNetworkUseInputAllProperties):
		globalFeatureNeuronsActivation = globalFeatureNeurons[arrayIndexPropertiesActivation]
		cDim = 2
	else:
		globalFeatureNeuronsActivation = globalFeatureNeurons
		cDim = 1
		
	globalFeatureNeuronsActivationAllSegments = pt.sum(globalFeatureNeuronsActivation, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 

	#topk column selection;
	if(inferencePredictiveNetworkModelFilterColumnsKmax):
		conceptColumnsActivation = GIAANNproto_sparseTensors.sparse_rowwise_max(globalFeatureNeuronsActivationAllSegments)
		#conceptColumnsActivation = pt.max(globalFeatureNeuronsActivationAllSegments, dim=1)	#max of all feature activations in columns	#only supports dense tensors
	else:
		conceptColumnsActivation = pt.sum(globalFeatureNeuronsActivationAllSegments, dim=1)	#sum across all feature activations in columns
	conceptColumnsActivation = conceptColumnsActivation.to_dense()	#convert to dense tensor (required for topk)
	conceptColumnsActivationTopkConcepts = pt.topk(conceptColumnsActivation, kc)
	conceptColumnsActivationTopkConceptsIndices = conceptColumnsActivationTopkConcepts.indices
	
	globalFeatureNeuronsFilteredColumns = GIAANNproto_sparseTensors.sliceSparseTensorMulti(globalFeatureNeurons, cDim, conceptColumnsActivationTopkConceptsIndices)	#select topk concept indices
	
	return globalFeatureNeuronsFilteredColumns, conceptColumnsActivationTopkConceptsIndices
