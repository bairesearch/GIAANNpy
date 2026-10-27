"""GIAANNnlp_predictionSourceEvidence.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP prediction Source Evidence

Training-support-dependent interpolation of exact and contextual source neurons.

"""

from collections import OrderedDict
from numbers import Integral
import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles


def initialiseSemanticSourceEvidence(databaseNetworkObject):
	if(inferenceInferMissingFeaturesSemanticAdaptiveSupplement):
		databaseNetworkObject.semanticSourceEvidenceFrequencies = OrderedDict()
	return


def shouldSupplementTrainedSemanticSource(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex):
	result = False
	if(inferenceInferMissingFeaturesSemanticAdaptiveSupplement):
		observations = getSemanticSourceObservations(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
		result = True
		if(inferenceInferMissingFeaturesSemanticAdaptiveSupplementSparseSources):
			result = observations < inferenceSemanticSourceEvidenceMaximumObservations
	else:
		raise RuntimeError(inferenceSemanticSourceEvidenceInvalidConfiguration)
	return result


def calculateSemanticSupplementWeight(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(inferenceInferMissingFeaturesSemanticAdaptiveSupplement):
		observations = getSemanticSourceObservations(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
		result = inferenceSemanticSourceEvidencePriorMass / (inferenceSemanticSourceEvidencePriorMass + observations)
		if(not math.isfinite(result) or not inferenceSemanticSourceEvidenceZero < result < inferenceSemanticSourceEvidenceOne):
			raise RuntimeError(inferenceSemanticSourceEvidenceInvalidWeight)
	else:
		raise RuntimeError(inferenceSemanticSourceEvidenceInvalidConfiguration)
	return result


def getSemanticSourceObservations(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(inferenceInferMissingFeaturesSemanticAdaptiveSupplement):
		if(databaseNetworkObject is None or not databaseNetworkObject.inferenceMode or not hasattr(databaseNetworkObject, inferenceSemanticSourceEvidenceCacheAttribute)):
			raise RuntimeError(inferenceSemanticSourceEvidenceInvalidState)
		for index, limit in ((sourceColumnIndex, databaseNetworkObject.c), (sourceFeatureIndex, databaseNetworkObject.f)):
			if(not isinstance(index, Integral) or isinstance(index, bool) or not inferenceSemanticSourceEvidenceZero <= index < limit):
				raise ValueError(inferenceSemanticSourceEvidenceInvalidNeuron)
		cache = databaseNetworkObject.semanticSourceEvidenceFrequencies
		if(sourceColumnIndex not in cache):
			cache[sourceColumnIndex] = loadSemanticSourceFrequencies(databaseNetworkObject, int(sourceColumnIndex))
			if(len(cache) > inferenceSemanticSourceEvidenceCacheColumns):
				cache.popitem(last=False)
		cache.move_to_end(sourceColumnIndex)
		keys, frequencies = cache[sourceColumnIndex]
		position = int(pt.searchsorted(keys, int(sourceFeatureIndex)).item())
		if(position >= keys.numel() or int(keys[position].item()) != sourceFeatureIndex):
			raise RuntimeError(inferenceSemanticSourceEvidenceMissingObservations)
		result = float(frequencies[position].item())
		if(not math.isfinite(result) or result <= inferenceSemanticSourceEvidenceZero):
			raise RuntimeError(inferenceSemanticSourceEvidenceMissingObservations)
	else:
		raise RuntimeError(inferenceSemanticSourceEvidenceInvalidConfiguration)
	return result


def loadSemanticSourceFrequencies(databaseNetworkObject, sourceColumnIndex):
	result = None
	if(inferenceInferMissingFeaturesSemanticAdaptiveSupplement):
		if(databaseNetworkObject is None or not databaseNetworkObject.inferenceMode):
			raise RuntimeError(inferenceSemanticSourceEvidenceInvalidState)
		if(not isinstance(sourceColumnIndex, Integral) or isinstance(sourceColumnIndex, bool) or not inferenceSemanticSourceEvidenceZero <= sourceColumnIndex < databaseNetworkObject.c):
			raise ValueError(inferenceSemanticSourceEvidenceInvalidNeuron)
		neurons = GIAANNcmn_databaseNetworkFiles.loadObservedColumnFeatureNeuronsTensor(databaseNetworkObject, sourceColumnIndex, targetDevice=inferenceSemanticSourceEvidenceDevice)
		if(not isinstance(neurons, pt.Tensor) or neurons.layout != pt.sparse_coo or neurons.sparse_dim() != inferenceSemanticSourceEvidenceNeuronRank or neurons.dim() != inferenceSemanticSourceEvidenceNeuronRank or not neurons.is_floating_point()):
			raise RuntimeError(inferenceSemanticSourceEvidenceInvalidFrequencies)
		shape = pt.tensor(neurons.shape, dtype=pt.long, device=neurons.device).unsqueeze(inferenceSemanticSourceEvidenceOne)
		if((neurons._indices() < inferenceSemanticSourceEvidenceZero).any() or (neurons._indices() >= shape).any()):
			raise RuntimeError(inferenceSemanticSourceEvidenceInvalidFrequencies)
		neurons = neurons.coalesce()
		indices = neurons.indices()
		mask = indices[inferenceSemanticSourceEvidencePropertyDimension] == databaseNetworkObject.arrayIndexPropertiesStrengthIndex
		values = neurons.values()[mask].to(pt.float64)
		featureIndices = indices[inferenceSemanticSourceEvidenceFeatureDimension, mask]
		if(not pt.isfinite(values).all() or (values < inferenceSemanticSourceEvidenceZero).any() or (featureIndices < inferenceSemanticSourceEvidenceZero).any() or (featureIndices >= databaseNetworkObject.f).any()):
			raise RuntimeError(inferenceSemanticSourceEvidenceInvalidFrequencies)
		keys, inverse = pt.unique(featureIndices, sorted=True, return_inverse=True)
		#Neuron frequencies are repeated across dendritic coordinates, not independent observations.
		frequencies = pt.zeros(keys.shape, dtype=pt.float64, device=neurons.device)
		frequencies.scatter_reduce_(inferenceSemanticSourceEvidenceVectorDimension, inverse, values, reduce=inferenceSemanticSourceEvidenceMaximumReduction)
		result = keys, frequencies
	else:
		raise RuntimeError(inferenceSemanticSourceEvidenceInvalidConfiguration)
	return result
