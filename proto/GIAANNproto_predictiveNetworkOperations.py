"""GIAANNproto_predictiveNetworkOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto predictive Network Operations

"""

import torch as pt
import torch.nn as nn
import torch.optim as optim


from GIAANNproto_globalDefs import *
	

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
