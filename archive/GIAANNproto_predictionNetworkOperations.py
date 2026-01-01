"""GIAANNproto_predictionNetworkOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto prediction Network Operations

"""

import torch as pt
import torch.nn as nn
import torch.optim as optim


from GIAANNproto_globalDefs import *
	

def normaliseSparseTensor(sparseTensor, independentInputChannels):
	#normalise from 0 to 1
	if(independentInputChannels):
		normalisedSparseTensorList = []
		for propertyIndex in range(arrayNumberOfProperties):
			sparseTensorProperty = sparseTensor[propertyIndex]
			sparseTensorProperty = sparseTensorProperty.coalesce()
			values = sparseTensorProperty.values()
			if values.numel() > 0: 
				minVals = values.min()
				maxVals = values.max()
				normalisedValues = (values - minVals) / (maxVals - minVals + 1e-8)
				normalisedSparseTensorProperty = pt.sparse_coo_tensor(sparseTensorProperty.indices(), normalisedValues, sparseTensorProperty.size(), device=sparseTensor.device)
			else:
				normalisedSparseTensorProperty = sparseTensorProperty
			normalisedSparseTensorList.append(normalisedSparseTensorProperty)
		normalisedSparseTensor = pt.stack(normalisedSparseTensorList, dim=0)
	else:
		sparseTensor = sparseTensor.coalesce()
		values = sparseTensor.values()
		if values.numel() > 0: 
			minVals = values.min()
			maxVals = values.max()
			normalisedValues = (values - minVals) / (maxVals - minVals + 1e-8)
			normalisedSparseTensor = pt.sparse_coo_tensor(sparseTensor.indices(), normalisedValues, sparseTensor.size(), device=sparseTensor.device)
		else:
			normalisedSparseTensor = sparseTensor
	return normalisedSparseTensor

def normaliseDenseTensor(tensor, dim=1):
	#normalise from 0 to 1
	reduce_dims = [d for d in range(tensor.ndim) if d > dim]
	minVals = tensor.amin(dim=reduce_dims, keepdim=True)
	maxVals = tensor.amax(dim=reduce_dims, keepdim=True)
	normalisedTensor = (tensor - minVals) / (maxVals - minVals + 1e-8)
	return normalisedTensor
	
if(inferencePredictiveNetworkIndependentFCpredictions):
	def getTopkPredictionsC(outputsC):
		with pt.no_grad():
			if multipleTargets:
				probsC = pt.sigmoid(outputsC)
			else:
				probsC = F.softmax(outputsC, dim=-1)

			topkValuesC, topkIndicesC = pt.topk(probsC, kcPred, dim=1)

		topkC = topkIndicesC[0]  # Assuming batch_size=1
		return topkC

	def getTopkPredictionsF(outputsF):
		with pt.no_grad():
			if multipleTargets:
				probsF = pt.sigmoid(outputsF)
			else:
				probsF = F.softmax(outputsF, dim=-1)

			topkValuesF, topkIndicesF = pt.topk(probsF, kf, dim=1)

		topkF = topkIndicesF[0]  # Assuming batch_size=1
		topkF = topkF.unsqueeze(-1)	#assume kf=1
		return topkF
else:
	def getTopkPredictions(outputs):
		with pt.no_grad():
			if(multipleTargets):
				# Apply sigmoid to get probabilities
				probs = pt.sigmoid(outputs)  # Shape: (batch_size, c, f)
			else:
				probs = outputs

			columnProbs = probs.mean(dim=2)  # Shape: (batch_size, c)
			_, conceptColumnsIndicesNext = pt.topk(columnProbs, kcNetwork, dim=1)

			# For each of the top kcNetwork columns, compute top kf features
			topKfIndices = []
			for columnIdx in conceptColumnsIndicesNext:
				columnData = probs[:, columnIdx, :]  # Shape: (batch_size, f)
				featureProbs = columnData.mean(dim=0)  # Shape: (f,)
				topkFeatureProbs, topkFeatureIndices = pt.topk(featureProbs, kf)  # Shapes: (kf,), (kf,)
				topKfIndices.append(topkFeatureIndices)

			conceptColumnsFeatureIndicesNext = pt.stack(topKfIndices)  # Shape: (batch_size, kcNetwork, kf)

		conceptColumnsIndicesNext = conceptColumnsIndicesNext[0]	#select first sample of batch
		conceptColumnsFeatureIndicesNext = conceptColumnsFeatureIndicesNext[0]	#select first sample of batch
		return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext


