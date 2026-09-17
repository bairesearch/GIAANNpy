"""Select a token by summing eligible LIF scores without changing neuron activations."""

from numbers import Integral
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_predictionConstraints


def selectLIFTokenAggregationCandidate(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState):
	result = None, None, None
	if(inferenceReviewPatch16TokenAggregation):
		validateLIFTokenAggregationCandidates(databaseNetworkObject, columnIndices, featureIndices, activationValues)
		#Connectivity, firing and deactivation filters have already run; apply the final node constraints before pooling.
		allowedMask = pt.tensor([GIAANNcmn_predictionConstraints.constraintAllowsNode(databaseNetworkObject, columnIndex, featureIndex, constraintState) and (inferenceColumnConstraintsAllowExternalPrimeConceptTransitionsBeamCandiateFilteringPatch or GIAANNcmn_predictionConstraints.constraintAllowsColumn(columnIndex, constraintState)) for columnIndex, featureIndex in zip(columnIndices.tolist(), featureIndices.tolist())], dtype=pt.bool, device=activationValues.device)
		columnIndices = columnIndices[allowedMask]
		featureIndices = featureIndices[allowedMask]
		activationValues = activationValues[allowedMask]
		if(activationValues.numel() > inferenceLIFTokenAggregationMinimumActivation):
			tokenIndices = featureIndices.clone()
			primeMask = featureIndices == featureIndexPrimeConceptNeuron
			if(primeMask.any()):
				primeTokens = [databaseNetworkObject.conceptColumnsList[columnIndex] for columnIndex in columnIndices[primeMask].tolist()]
				if(any(token not in databaseNetworkObject.conceptFeaturesDict for token in primeTokens)):
					raise RuntimeError(inferenceLIFTokenAggregationInvalidToken)
				primeTokenIndices = [databaseNetworkObject.conceptFeaturesDict[token] for token in primeTokens]
				if(any(not isinstance(tokenIndex, Integral) or isinstance(tokenIndex, bool) or tokenIndex < inferenceLIFTokenAggregationMinimumActivation or tokenIndex >= databaseNetworkObject.f or databaseNetworkObject.conceptFeaturesList[tokenIndex] != token for tokenIndex, token in zip(primeTokenIndices, primeTokens))):
					raise RuntimeError(inferenceLIFTokenAggregationInvalidToken)
				tokenIndices[primeMask] = pt.tensor(primeTokenIndices, dtype=tokenIndices.dtype, device=tokenIndices.device)
			uniqueTokens, inverseIndices = pt.unique(tokenIndices, sorted=True, return_inverse=True)
			tokenTotals = pt.zeros_like(uniqueTokens, dtype=activationValues.dtype)
			tokenTotals.scatter_add_(inferenceLIFTokenAggregationVectorDimension, inverseIndices, activationValues)
			if(not pt.isfinite(tokenTotals).all() or (tokenTotals <= inferenceLIFTokenAggregationMinimumActivation).any()):
				raise RuntimeError(inferenceLIFTokenAggregationInvalidTotals)
			winningTokenIndex = pt.topk(tokenTotals, inferenceLIFTokenAggregationSelectionCount).indices
			#Select an existing neuron of the winning token; retain its original score and the original top-k tie behaviour.
			winningTokenActivations = pt.where(inverseIndices == winningTokenIndex, activationValues, inferenceLIFTokenAggregationExcludedActivation)
			winningNeuronIndex = pt.topk(winningTokenActivations, inferenceLIFTokenAggregationSelectionCount).indices
			result = columnIndices.index_select(inferenceLIFTokenAggregationVectorDimension, winningNeuronIndex), featureIndices.index_select(inferenceLIFTokenAggregationVectorDimension, winningNeuronIndex), activationValues.index_select(inferenceLIFTokenAggregationVectorDimension, winningNeuronIndex)
	else:
		raise RuntimeError(inferenceLIFTokenAggregationInvalidConfiguration)
	return result


def validateLIFTokenAggregationCandidates(databaseNetworkObject, columnIndices, featureIndices, activationValues):
	if(inferenceReviewPatch16TokenAggregation):
		if(not inferenceLeakyIntegrateAndFire or not predictionEnsureConnectedToPreviousPrediction or not enforceDirectConnectionsSANI or inferenceBeamSearch or algorithmMatrixSANIenforceRequirement != inferenceLIFTokenAggregationRequiredSANICondition):
			raise RuntimeError(inferenceLIFTokenAggregationInvalidConfiguration)
		if(databaseNetworkObject is None or not databaseNetworkObject.inferenceMode or len(databaseNetworkObject.conceptColumnsList) != databaseNetworkObject.c or len(databaseNetworkObject.conceptFeaturesList) != databaseNetworkObject.f or not isinstance(databaseNetworkObject.conceptFeaturesDict, dict)):
			raise RuntimeError(inferenceLIFTokenAggregationInvalidDatabase)
		if(not all(isinstance(vector, pt.Tensor) and vector.dim() == inferenceLIFTokenAggregationCandidateRank for vector in (columnIndices, featureIndices, activationValues)) or columnIndices.shape != activationValues.shape or featureIndices.shape != activationValues.shape or not activationValues.is_floating_point() or columnIndices.dtype != pt.long or featureIndices.dtype != pt.long or columnIndices.device != activationValues.device or featureIndices.device != activationValues.device):
			raise RuntimeError(inferenceLIFTokenAggregationInvalidCandidates)
		if(not pt.isfinite(activationValues).all() or (activationValues <= inferenceLIFTokenAggregationMinimumActivation).any() or (columnIndices < inferenceLIFTokenAggregationMinimumActivation).any() or (columnIndices >= databaseNetworkObject.c).any() or (featureIndices < inferenceLIFTokenAggregationMinimumActivation).any() or (featureIndices >= databaseNetworkObject.f).any()):
			raise RuntimeError(inferenceLIFTokenAggregationInvalidCandidates)
	else:
		raise RuntimeError(inferenceLIFTokenAggregationInvalidConfiguration)
	return
