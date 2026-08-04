"""GIAANNnlp_auxiliaryNeuronsPOS.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP deterministic POS auxiliary source features

"""

from GIAANNcmn_globalDefs import *


if(auxiliaryNeurons and auxiliaryNeuronsPOS):

	def getTokenPOSAuxiliaryFeatureIndices(databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures=False, registerParent=False):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		result = []
		if(not isinstance(isConcept, bool)):
			raise RuntimeError("getTokenPOSAuxiliaryFeatureIndices error: isConcept must be bool")
		if(not isinstance(allowNewFeatures, bool)):
			raise RuntimeError("getTokenPOSAuxiliaryFeatureIndices error: allowNewFeatures must be bool")
		if(not isinstance(registerParent, bool)):
			raise RuntimeError("getTokenPOSAuxiliaryFeatureIndices error: registerParent must be bool")
		if(not isConcept and tokenHasPOSAuxiliaryParentWord(token)):
			parentWord = getTokenPOSAuxiliaryParentFeatureValue(databaseNetworkObject, token)
			for auxiliaryFeaturePrefix, auxiliaryFeatureValue in createTokenPOSAuxiliaryFeatureRecords(token):
				auxiliaryFeatureWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryFeatureValue)
				auxiliaryFeatureIndex = GIAANNnlp_auxiliaryNeuronsSimilarWords.registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, allowNewFeatures)
				if(registerParent):
					parentKey = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildSimilarityParentKey(auxiliaryFeaturePrefix, parentWord)
					activationWeight = getPOSAuxiliaryFeatureActivationWeight(auxiliaryFeaturePrefix)
					GIAANNnlp_auxiliaryNeuronsSimilarWords.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, auxiliaryFeatureWord, activationWeight)
				result.append(auxiliaryFeatureIndex)
		return result

	def calculatePOSAuxiliaryFeatureActivations(databaseNetworkObject, conceptIndex, sourceFeatureIndex, sourceActivationValue, targetDevice):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		if(not isinstance(sourceActivationValue, pt.Tensor)):
			raise RuntimeError("calculatePOSAuxiliaryFeatureActivations error: sourceActivationValue must be a tensor")
		if(sourceActivationValue.numel() != 1):
			raise RuntimeError("calculatePOSAuxiliaryFeatureActivations error: sourceActivationValue must contain one value")
		parentFeatureValue = getPOSAuxiliaryParentFeatureValueForSourceFeatureIndex(databaseNetworkObject, sourceFeatureIndex)
		result = pt.zeros((databaseNetworkObject.fas,), dtype=arrayType, device=targetDevice)
		sourceActivationValueTarget = sourceActivationValue.to(targetDevice)
		for auxiliaryFeaturePrefix in GIAANNnlp_auxiliaryNeuronsSimilarWords.getPOSAuxiliaryFeaturePrefixes():
			parentKey = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildSimilarityParentKey(auxiliaryFeaturePrefix, parentFeatureValue)
			similarityThreshold = GIAANNnlp_auxiliaryNeuronsSimilarWords.getSimilarityThresholdForAuxiliaryFeaturePrefix(auxiliaryFeaturePrefix)
			activationRecords = GIAANNnlp_auxiliaryNeuronsSimilarWords.getAuxiliaryFeatureActivationRecordsForParentKeyAndConceptIndex(databaseNetworkObject, parentKey, conceptIndex)
			for auxiliaryConceptIndex, auxiliaryFeatureIndex, activationWeight in activationRecords:
				if(activationWeight >= similarityThreshold):
					result[auxiliaryFeatureIndex] = sourceActivationValueTarget * activationWeight
		return result

	def getTokenPOSAuxiliaryParentFeatureValue(databaseNetworkObject, token):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		result = None
		if(tokeniserSubword and useDedicatedFeatureListsSubword):
			import GIAANNnlp_sequenceTokens
			sourceFeatureIndex = GIAANNnlp_sequenceTokens.getTokeniserSubwordFeatureIndex(token)
			if(sourceFeatureIndex <= featureIndexPrimeConceptNeuron or sourceFeatureIndex >= databaseNetworkObject.f):
				raise RuntimeError("getTokenPOSAuxiliaryParentFeatureValue error: source feature index out of range")
			result = auxiliaryNeuronsPOSParentFeatureIndexPrefix + str(sourceFeatureIndex)
		else:
			result = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(token.word)
		return result

	def getPOSAuxiliaryParentFeatureValueForSourceFeatureIndex(databaseNetworkObject, sourceFeatureIndex):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		result = None
		if(isinstance(sourceFeatureIndex, bool)):
			raise RuntimeError("getPOSAuxiliaryParentFeatureValueForSourceFeatureIndex error: sourceFeatureIndex must not be bool")
		normalisedSourceFeatureIndex = int(sourceFeatureIndex)
		if(normalisedSourceFeatureIndex <= featureIndexPrimeConceptNeuron or normalisedSourceFeatureIndex >= databaseNetworkObject.f):
			raise RuntimeError("getPOSAuxiliaryParentFeatureValueForSourceFeatureIndex error: source feature index out of range")
		if(tokeniserSubword and useDedicatedFeatureListsSubword):
			result = auxiliaryNeuronsPOSParentFeatureIndexPrefix + str(normalisedSourceFeatureIndex)
		else:
			if(normalisedSourceFeatureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
				raise RuntimeError("getPOSAuxiliaryParentFeatureValueForSourceFeatureIndex error: source feature name is unavailable")
			result = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(databaseNetworkObject.conceptFeaturesList[normalisedSourceFeatureIndex])
		return result

	def getPOSAuxiliarySourceFeatureIndex(databaseNetworkObject, parentFeatureValue):
		result = None
		if(tokeniserSubword and useDedicatedFeatureListsSubword):
			if(not parentFeatureValue.startswith(auxiliaryNeuronsPOSParentFeatureIndexPrefix)):
				raise RuntimeError("getPOSAuxiliarySourceFeatureIndex error: parent feature value has no feature-index prefix")
			featureIndexText = parentFeatureValue[len(auxiliaryNeuronsPOSParentFeatureIndexPrefix):]
			if(not featureIndexText.isdigit()):
				raise RuntimeError("getPOSAuxiliarySourceFeatureIndex error: parent feature index is invalid")
			result = int(featureIndexText)
			if(result <= featureIndexPrimeConceptNeuron or result >= databaseNetworkObject.f):
				raise RuntimeError("getPOSAuxiliarySourceFeatureIndex error: parent feature index out of range")
		else:
			if(parentFeatureValue not in databaseNetworkObject.conceptFeaturesDict):
				raise RuntimeError("getPOSAuxiliarySourceFeatureIndex error: parent feature is not registered")
			result = databaseNetworkObject.conceptFeaturesDict[parentFeatureValue]
		return result

	def getPOSAuxiliaryFeatureActivationWeight(auxiliaryFeaturePrefix):
		result = None
		if(auxiliaryFeaturePrefix == auxiliaryNeuronsPOSFeatureNamePrefixLemma):
			result = auxiliaryNeuronsPOSLemmaActivationWeight
		elif(auxiliaryFeaturePrefix == auxiliaryNeuronsPOSFeatureNamePrefixPartOfSpeech):
			result = auxiliaryNeuronsPOSPartOfSpeechActivationWeight
		elif(auxiliaryFeaturePrefix == auxiliaryNeuronsPOSFeatureNamePrefixSubwordRole):
			result = auxiliaryNeuronsPOSSubwordRoleActivationWeight
		else:
			raise RuntimeError("getPOSAuxiliaryFeatureActivationWeight error: unsupported auxiliary feature prefix")
		return result

	def createTokenPOSAuxiliaryFeatureRecords(token):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		result = []
		parentTokenComplete = isTokenPOSAuxiliaryParentTokenComplete(token)
		if(auxiliaryNeuronsPOSlemma and parentTokenComplete and tokenHasPOSAuxiliaryLemma(token)):
			lemma = getTokenPOSAuxiliaryLemma(token)
			result.append((auxiliaryNeuronsPOSFeatureNamePrefixLemma, GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(lemma)))
		if(auxiliaryNeuronsPOSpartOfSpeech and parentTokenComplete and tokenHasPOSAuxiliaryPartOfSpeech(token)):
			partOfSpeech = getTokenPOSAuxiliaryPartOfSpeech(token)
			result.append((auxiliaryNeuronsPOSFeatureNamePrefixPartOfSpeech, GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(partOfSpeech)))
		if(auxiliaryNeuronsPOSsubwordRole):
			subwordRole = getTokenPOSAuxiliarySubwordRole(token)
			result.append((auxiliaryNeuronsPOSFeatureNamePrefixSubwordRole, GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(subwordRole)))
		return result

	def tokenHasPOSAuxiliaryParentWord(token):
		if(not hasattr(token, "word")):
			raise RuntimeError("tokenHasPOSAuxiliaryParentWord error: token has no word")
		if(token.word is None):
			raise RuntimeError("tokenHasPOSAuxiliaryParentWord error: token word is None")
		result = str(token.word).strip() != auxiliaryNeuronsSimilarWordsFeatureValueEmpty
		return result

	def isTokenPOSAuxiliaryParentTokenComplete(token):
		result = True
		if(tokeniserSubword):
			subwordRole = getTokenPOSAuxiliarySubwordRole(token)
			result = subwordRole in auxiliaryNeuronsPOSCompletedSubwordRoleList
		return result

	def tokenHasPOSAuxiliaryLemma(token):
		lemma = None
		if(tokeniserSubword):
			if(not hasattr(token, "posLemma")):
				raise RuntimeError("tokenHasPOSAuxiliaryLemma error: subword token has no posLemma")
			lemma = token.posLemma
		else:
			if(not hasattr(token, "lemma")):
				raise RuntimeError("tokenHasPOSAuxiliaryLemma error: token has no lemma")
			lemma = token.lemma
		result = lemma is not None and str(lemma).strip() != auxiliaryNeuronsSimilarWordsFeatureValueEmpty
		return result

	def tokenHasPOSAuxiliaryPartOfSpeech(token):
		partOfSpeech = None
		if(tokeniserSubword):
			if(not hasattr(token, "parentPos")):
				raise RuntimeError("tokenHasPOSAuxiliaryPartOfSpeech error: subword token has no parentPos")
			partOfSpeech = token.parentPos
		else:
			if(not hasattr(token, "pos")):
				raise RuntimeError("tokenHasPOSAuxiliaryPartOfSpeech error: token has no pos")
			partOfSpeech = token.pos
		result = partOfSpeech is not None and str(partOfSpeech).strip() != auxiliaryNeuronsSimilarWordsFeatureValueEmpty
		return result

	def getTokenPOSAuxiliaryLemma(token):
		result = None
		if(tokeniserSubword):
			if(not hasattr(token, "posLemma")):
				raise RuntimeError("getTokenPOSAuxiliaryLemma error: subword token has no posLemma")
			result = token.posLemma
		else:
			if(not hasattr(token, "lemma")):
				raise RuntimeError("getTokenPOSAuxiliaryLemma error: token has no lemma")
			result = token.lemma
		if(result is None or str(result).strip() == auxiliaryNeuronsSimilarWordsFeatureValueEmpty):
			raise RuntimeError("getTokenPOSAuxiliaryLemma error: token lemma is empty")
		return result

	def getTokenPOSAuxiliaryPartOfSpeech(token):
		result = None
		if(tokeniserSubword):
			if(not hasattr(token, "parentPos")):
				raise RuntimeError("getTokenPOSAuxiliaryPartOfSpeech error: subword token has no parentPos")
			result = token.parentPos
		else:
			if(not hasattr(token, "pos")):
				raise RuntimeError("getTokenPOSAuxiliaryPartOfSpeech error: token has no pos")
			result = token.pos
		if(result is None or str(result).strip() == auxiliaryNeuronsSimilarWordsFeatureValueEmpty):
			raise RuntimeError("getTokenPOSAuxiliaryPartOfSpeech error: token POS is empty")
		return result

	def getTokenPOSAuxiliarySubwordRole(token):
		if(not tokeniserSubword):
			raise RuntimeError("getTokenPOSAuxiliarySubwordRole error: tokeniserSubword is required")
		if(not hasattr(token, "subwordRole")):
			raise RuntimeError("getTokenPOSAuxiliarySubwordRole error: subword token has no subwordRole")
		result = token.subwordRole
		if(result is None or str(result).strip() == auxiliaryNeuronsSimilarWordsFeatureValueEmpty):
			raise RuntimeError("getTokenPOSAuxiliarySubwordRole error: subword role is empty")
		return result
