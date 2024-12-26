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
	def getTopkPredictionsC(outputs_c):
		with pt.no_grad():
			if multipleTargets:
				probs_c = pt.sigmoid(outputs_c)
			else:
				probs_c = F.softmax(outputs_c, dim=-1)

			topk_values_c, topk_indices_c = pt.topk(probs_c, kcPred, dim=1)

		topk_c = topk_indices_c[0]  # Assuming batch_size=1
		return topk_c

	def getTopkPredictionsF(outputs_f):
		with pt.no_grad():
			if multipleTargets:
				probs_f = pt.sigmoid(outputs_f)
			else:
				probs_f = F.softmax(outputs_f, dim=-1)

			topk_values_f, topk_indices_f = pt.topk(probs_f, kf, dim=1)

		topk_f = topk_indices_f[0]  # Assuming batch_size=1
		topk_f = topk_f.unsqueeze(-1)	#assume kf=1
		return topk_f
else:
	def getTopkPredictions(outputs):
		with pt.no_grad():
			if(multipleTargets):
				# Apply sigmoid to get probabilities
				probs = pt.sigmoid(outputs)  # Shape: (batch_size, c, f)
			else:
				probs = outputs

			column_probs = probs.mean(dim=2)  # Shape: (batch_size, c)
			_, concept_columns_indices_next = pt.topk(column_probs, kcNetwork, dim=1)

			# For each of the top kcNetwork columns, compute top kf features
			top_kf_indices = []
			for column_idx in concept_columns_indices_next:
				column_data = probs[:, column_idx, :]  # Shape: (batch_size, f)
				feature_probs = column_data.mean(dim=0)  # Shape: (f,)
				topk_feature_probs, topk_feature_indices = pt.topk(feature_probs, kf)  # Shapes: (kf,), (kf,)
				top_kf_indices.append(topk_feature_indices)

			concept_columns_feature_indices_next = pt.stack(top_kf_indices)  # Shape: (batch_size, kcNetwork, kf)

		concept_columns_indices_next = concept_columns_indices_next[0]	#select first sample of batch
		concept_columns_feature_indices_next = concept_columns_feature_indices_next[0]	#select first sample of batch
		return concept_columns_indices_next, concept_columns_feature_indices_next
