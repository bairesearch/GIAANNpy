"""GIAANNnlp_predictionSemantic.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP prediction Semantic

Selected-token-conditioned auxiliary propagation using a frozen trained database.

"""


import os
import math
from numbers import Integral
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_predictionActivate
import GIAANNcmn_predictionConstraints
if(inferenceInferMissingFeaturesSemanticAdaptiveSupplement):
	import GIAANNnlp_predictionSourceEvidence


def setSemanticObservedSource(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex):
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic):
		#The caller supplies the selected source, including generated tokens during prediction-driven inference.
		validateSemanticConfiguration(databaseNetworkObject)
		if(not isinstance(sourceColumnIndex, Integral) or isinstance(sourceColumnIndex, bool) or not isinstance(sourceFeatureIndex, Integral) or isinstance(sourceFeatureIndex, bool) or sourceColumnIndex < arrayIndexSegmentFirst or sourceColumnIndex >= databaseNetworkObject.c or sourceFeatureIndex < arrayIndexSegmentFirst or sourceFeatureIndex >= databaseNetworkObject.f):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidState)
		databaseNetworkObject.inferenceSemanticObservedSource = sourceColumnIndex, sourceFeatureIndex
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return


def getSemanticMissingFeatureSources(databaseNetworkObject, globalFeatureNeuronsActivation):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic):
		validateSemanticConfiguration(databaseNetworkObject)
		sourceColumnIndex, sourceFeatureIndex = databaseNetworkObject.inferenceSemanticObservedSource
		sourceToken = getSemanticSourceToken(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
		if(not isinstance(globalFeatureNeuronsActivation, pt.Tensor) or not globalFeatureNeuronsActivation.is_sparse or tuple(globalFeatureNeuronsActivation.shape) != (multipleDendriticBranchesNumber, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f) or not pt.isfinite(globalFeatureNeuronsActivation._values()).all() or (globalFeatureNeuronsActivation._values() < inferenceInferMissingFeaturesMinimumActivation).any()):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidState)
		soma = GIAANNcmn_predictionActivate.calculateLeakyIntegrateAndFireSomaActivation(globalFeatureNeuronsActivation).coalesce()
		if(not pt.isfinite(soma.values()).all() or (soma.values() < inferenceInferMissingFeaturesMinimumActivation).any()):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidState)
		candidates = {}
		sameFeatureIndex = databaseNetworkObject.conceptFeaturesDict.get(sourceToken)
		sameColumnIndex = databaseNetworkObject.conceptColumnsDict.get(sourceToken)
		identityMask = pt.zeros_like(soma.values(), dtype=pt.bool)
		if(sameFeatureIndex is not None and sameFeatureIndex != featureIndexPrimeConceptNeuron):
			identityMask |= soma.indices()[inferenceLeakyIntegrateAndFireSomaActivationFeatureDimension] == sameFeatureIndex
		if(sameColumnIndex is not None):
			identityMask |= (soma.indices()[inferenceLeakyIntegrateAndFireSomaActivationConceptDimension] == sameColumnIndex) & (soma.indices()[inferenceLeakyIntegrateAndFireSomaActivationFeatureDimension] == featureIndexPrimeConceptNeuron)
		identityIndices = pt.nonzero(identityMask & (soma.values() > inferenceInferMissingFeaturesMinimumActivation), as_tuple=False).flatten()
		for candidateIndex in identityIndices.tolist():
			columnIndex, featureIndex = soma.indices()[:, candidateIndex].tolist()
			candidates[columnIndex, featureIndex] = float(soma.values()[candidateIndex].item())
		if(inferenceInferMissingFeaturesSemanticContextColumns and sameFeatureIndex is not None and sameFeatureIndex != featureIndexPrimeConceptNeuron):
			activation = globalFeatureNeuronsActivation.coalesce()
			columnScores = pt.zeros((databaseNetworkObject.c,), dtype=activation.dtype, device=activation.device)
			columnScores.scatter_reduce_(arrayIndexSegmentFirst, activation.indices()[inferenceInferMissingFeaturesConceptDimension], activation.values(), reduce=inferenceInferMissingFeaturesSemanticMaximumReduction)
			activeContextIndices = pt.nonzero(columnScores > inferenceInferMissingFeaturesMinimumActivation, as_tuple=False).flatten()
			contextIndices = activeContextIndices[pt.argsort(columnScores[activeContextIndices], descending=True, stable=True)[:inferenceInferMissingFeaturesSemanticContextTopK]]
			for columnIndex in contextIndices.tolist():
				columnScore = float(columnScores[columnIndex].item())
				if(columnScore > inferenceInferMissingFeaturesMinimumActivation):
					key = columnIndex, sameFeatureIndex
					candidates[key] = max(candidates.get(key, inferenceInferMissingFeaturesMinimumActivation), columnScore)
		if(inferenceInferMissingFeaturesSemanticWordVectors):
			vectorIndices = pt.argsort(soma.values(), descending=True, stable=True)[:inferenceInferMissingFeaturesSemanticVectorTopK]
			for candidateIndex in vectorIndices.tolist():
				columnIndex, featureIndex = soma.indices()[:, candidateIndex].tolist()
				candidateToken = getSemanticSourceToken(databaseNetworkObject, columnIndex, featureIndex)
				similarity = calculateSemanticTokenSimilarity(databaseNetworkObject, sourceToken, candidateToken)
				if(similarity >= inferenceInferMissingFeaturesSemanticVectorThreshold):
					key = columnIndex, featureIndex
					candidateScore = float(soma.values()[candidateIndex].item()) * similarity ** inferenceInferMissingFeaturesSemanticVectorExponent
					candidates[key] = max(candidates.get(key, inferenceInferMissingFeaturesMinimumActivation), candidateScore)
		trainedCandidates = []
		for (columnIndex, featureIndex), score in sorted(candidates.items(), key=lambda item: (-item[inferenceInferMissingFeaturesSemanticDictionaryValueField], item[inferenceInferMissingFeaturesSemanticDictionaryKeyField])):
			if(score > inferenceInferMissingFeaturesMinimumActivation and (columnIndex, featureIndex) != (sourceColumnIndex, sourceFeatureIndex) and semanticSourceConnectionsExist(databaseNetworkObject, columnIndex, featureIndex)):
				trainedCandidates.append((columnIndex, featureIndex, score))
				if(len(trainedCandidates) == inferenceInferMissingFeaturesSemanticTopK):
					break
		if(inferenceInferMissingFeaturesSemanticSupplementTrainedSources and databaseNetworkObject.inferenceSemanticObservedSourceTrained):
			donorTotal = sum(candidate[inferenceInferMissingFeaturesSemanticSourceScoreField] for candidate in trainedCandidates)
			observedWeight = inferenceInferMissingFeaturesNormalisedActivationTotal
			if(donorTotal > inferenceInferMissingFeaturesMinimumActivation):
				if(inferenceInferMissingFeaturesSemanticAdaptiveSupplement):
					supplementWeight = GIAANNnlp_predictionSourceEvidence.calculateSemanticSupplementWeight(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
					trainedCandidates = [(columnIndex, featureIndex, score * supplementWeight / donorTotal) for columnIndex, featureIndex, score in trainedCandidates]
					observedWeight -= supplementWeight
				else:
					trainedCandidates = [(columnIndex, featureIndex, score * inferenceInferMissingFeaturesSemanticSupplementWeight / donorTotal) for columnIndex, featureIndex, score in trainedCandidates]
					observedWeight -= inferenceInferMissingFeaturesSemanticSupplementWeight
			trainedCandidates.insert(arrayIndexSegmentFirst, (sourceColumnIndex, sourceFeatureIndex, observedWeight))
		columns = [candidate[inferenceInferMissingFeaturesSemanticSourceColumnField] for candidate in trainedCandidates]
		features = [candidate[inferenceInferMissingFeaturesSemanticSourceFeatureField] for candidate in trainedCandidates]
		scores = pt.tensor([candidate[inferenceInferMissingFeaturesSemanticSourceScoreField] for candidate in trainedCandidates], dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
		confidence = pt.full((), inferenceInferMissingFeaturesNormalisedActivationTotal, dtype=scores.dtype, device=scores.device)
		result = columns, features, list(scores.unbind()), list(scores.unbind()), scores.sum(), confidence
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return result


def projectSemanticDonorTargets(databaseNetworkObject, targetActivation, donorColumnIndex):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic and inferenceInferMissingFeaturesSemanticProjectColumns):
		validateSemanticConfiguration(databaseNetworkObject)
		if(not targetActivation.is_sparse or targetActivation.dim() != inferenceInferMissingFeaturesNeuronTensorRank or donorColumnIndex < arrayIndexSegmentFirst or donorColumnIndex >= databaseNetworkObject.c):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidState)
		sourceColumnIndex, sourceFeatureIndex = databaseNetworkObject.inferenceSemanticObservedSource
		activation = targetActivation.coalesce()
		indices = activation.indices().clone()
		indices[inferenceInferMissingFeaturesConceptDimension, indices[inferenceInferMissingFeaturesConceptDimension] == donorColumnIndex] = sourceColumnIndex
		result = pt.sparse_coo_tensor(indices, activation.values(), size=activation.size(), dtype=activation.dtype, device=activation.device).coalesce()
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return result


def buildSemanticDonorConstraints(databaseNetworkObject, observedColumnsDict, donorColumns, donorFeatures, connectedColumnsConstraint):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic and inferenceInferMissingFeaturesSemanticProjectColumns):
		validateSemanticConfiguration(databaseNetworkObject)
		if(len(donorColumns) != len(donorFeatures) or not donorColumns):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidState)
		sourceColumnIndex, sourceFeatureIndex = databaseNetworkObject.inferenceSemanticObservedSource
		columns = set()
		features = {}
		featureDetailsAvailable = False
		for donorColumn, donorFeature in zip(donorColumns, donorFeatures):
			donorConnectedColumns, donorConnectedFeatures = GIAANNcmn_predictionConstraints.buildConnectedColumnsLookup(databaseNetworkObject, observedColumnsDict, [(donorColumn, donorFeature)], connectedColumnsConstraint.device, connectedColumnsConstraint.dtype)
			for targetColumn in donorConnectedColumns.tolist():
				projectedColumn = sourceColumnIndex if targetColumn == donorColumn else targetColumn
				columns.add(projectedColumn)
				if(donorConnectedFeatures is not None):
					features.setdefault(projectedColumn, set()).update(donorConnectedFeatures.get(targetColumn, set()))
			featureDetailsAvailable = featureDetailsAvailable or donorConnectedFeatures is not None
		result = pt.tensor(sorted(columns), dtype=connectedColumnsConstraint.dtype, device=connectedColumnsConstraint.device), features if featureDetailsAvailable else None
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return result


def getSemanticSourceToken(databaseNetworkObject, columnIndex, featureIndex):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic):
		if(not isinstance(columnIndex, Integral) or isinstance(columnIndex, bool) or not isinstance(featureIndex, Integral) or isinstance(featureIndex, bool) or columnIndex < arrayIndexSegmentFirst or columnIndex >= databaseNetworkObject.c or featureIndex < arrayIndexSegmentFirst or featureIndex >= databaseNetworkObject.f):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidState)
		result = databaseNetworkObject.conceptColumnsList[columnIndex] if featureIndex == featureIndexPrimeConceptNeuron else databaseNetworkObject.conceptFeaturesList[featureIndex]
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return result


def calculateSemanticTokenSimilarity(databaseNetworkObject, sourceToken, candidateToken):
	result = inferenceInferMissingFeaturesMinimumActivation
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic and inferenceInferMissingFeaturesSemanticWordVectors):
		if(sourceToken == candidateToken):
			result = inferenceInferMissingFeaturesNormalisedActivationTotal
		elif(sourceToken.strip().isalpha() and candidateToken.strip().isalpha()):
			if(databaseNetworkObject.inferenceSemanticVectorVocabulary is None):
				import spacy
				vectorModel = spacy.load(inferenceInferMissingFeaturesSemanticVectorModel)
				if(vectorModel.vocab.vectors_length <= inferenceInferMissingFeaturesMinimumActivation):
					raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidConfiguration)
				databaseNetworkObject.inferenceSemanticVectorVocabulary = vectorModel.vocab
			vocabulary = databaseNetworkObject.inferenceSemanticVectorVocabulary
			sourceLexeme = vocabulary[sourceToken.strip().lower()]
			candidateLexeme = vocabulary[candidateToken.strip().lower()]
			if(sourceLexeme.has_vector and candidateLexeme.has_vector and sourceLexeme.vector_norm > inferenceInferMissingFeaturesMinimumActivation and candidateLexeme.vector_norm > inferenceInferMissingFeaturesMinimumActivation):
				similarity = float(sourceLexeme.vector.dot(candidateLexeme.vector) / (sourceLexeme.vector_norm * candidateLexeme.vector_norm))
				if(not math.isfinite(similarity)):
					raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidState)
				result = max(inferenceInferMissingFeaturesMinimumActivation, min(inferenceInferMissingFeaturesNormalisedActivationTotal, similarity))
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return result


def semanticSourceConnectionsExist(databaseNetworkObject, columnIndex, featureIndex):
	result = False
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic):
		getSemanticSourceToken(databaseNetworkObject, columnIndex, featureIndex)
		folder = GIAANNcmn_databaseNetworkFiles.getObservedColumnFeatureConnectionsFolder(columnIndex)
		filename = GIAANNcmn_databaseNetworkFiles.getObservedColumnSourceFeatureConnectionsFileBaseName(featureIndex) + pytorchTensorFileExtension
		result = GIAANNcmn_databaseNetworkFiles.pathExists(os.path.join(folder, filename))
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return result


def validateSemanticConfiguration(databaseNetworkObject):
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesSemantic):
		if(not useModalityNLP or not inferenceLeakyIntegrateAndFire or not inferenceInferMissingFeaturesUpdate7UsePersistedSourceConnections or useTrainDuringInference or inferenceBeamSearch or databaseNetworkObject is None or not databaseNetworkObject.inferenceMode or (inferenceInferMissingFeaturesSemanticSupplementTrainedSources and (not inferenceInferMissingFeaturesSemanticObservedBurst or inferenceInferMissingFeaturesSemanticSupplementWeight >= inferenceInferMissingFeaturesNormalisedActivationTotal))):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidConfiguration)
		if(inferenceInferMissingFeaturesUpdate1SelectTopKCandidates or inferenceInferMissingFeaturesUpdate2SourceConfidence or inferenceInferMissingFeaturesUpdate5SelectTopKConstraintSources or not inferenceInferMissingFeaturesUpdate6RetainCandidateActivations or not inferenceInferMissingFeaturesUpdate8RequireActiveProxySignal):
			raise RuntimeError(inferenceInferMissingFeaturesSemanticInvalidLegacyOptions)
	else:
		raise RuntimeError(inferenceInferMissingFeaturesSemanticDisabledError)
	return
