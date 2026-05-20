"""GIAANNnlp_auxiliaryNeuronsSubword.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP auxiliary neurons subword

"""

import os
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles


if(auxiliaryNeuronsTokenisationSubword):

	auxiliaryNeuronsTokenisationSubwordSavedSourceTensorPaths = set()

	def getPreprocessedTokenMorphString(preprocessedToken):
		result = auxiliaryNeuronsTokenisationSubwordMorphEmpty
		if(hasattr(preprocessedToken, "morph")):
			result = str(preprocessedToken.morph)
		elif(hasattr(preprocessedToken, "morph_")):
			result = str(preprocessedToken.morph_)
		if(result is None):
			result = auxiliaryNeuronsTokenisationSubwordMorphEmpty
		return result

	def createTokenAuxiliaryFeatureWords(token):
		result = []
		if(shouldCreateTokenBaseAuxiliaryFeature(token)):
			result.append(buildAuxiliaryFeatureName(getTokenAuxiliaryBaseForm(token)))
		return result

	def shouldCreateTokenBaseAuxiliaryFeature(token):
		baseForm = getTokenAuxiliaryBaseForm(token)
		distinctForm = isTokenAuxiliaryDistinctForm(token, baseForm)
		result = False
		if(baseForm != auxiliaryNeuronsTokenisationSubwordFeatureValueEmpty):
			if(auxiliaryNeuronsTokenisationSubwordDistinctEnforce and not distinctForm):
				result = False
			else:
				if(auxiliaryNeuronsTokenisationSubwordMorph and tokenMorphIndicatesAuxiliaryVariation(token)):
					result = True
				if((not result) and auxiliaryNeuronsTokenisationSubwordSuffix and tokenSuffixIndicatesAuxiliaryVariation(token)):
					result = True
				if((not result) and auxiliaryNeuronsTokenisationSubwordLemma and distinctForm):
					result = True
		return result

	def getTokenAuxiliaryBaseForm(token):
		if(token.lemma is None):
			raise RuntimeError("getTokenAuxiliaryBaseForm error: token lemma is None")
		if(token.word is None):
			raise RuntimeError("getTokenAuxiliaryBaseForm error: token word is None")
		result = token.lemma
		return result

	def isTokenAuxiliaryDistinctForm(token, baseForm):
		if(token.word is None):
			raise RuntimeError("isTokenAuxiliaryDistinctForm error: token word is None")
		result = baseForm != token.word
		return result

	def tokenMorphIndicatesAuxiliaryVariation(token):
		morphString = token.morph
		if(morphString is None):
			morphString = auxiliaryNeuronsTokenisationSubwordMorphEmpty
		result = False
		if(morphString != auxiliaryNeuronsTokenisationSubwordMorphEmpty):
			morphParts = morphString.split(auxiliaryNeuronsTokenisationSubwordMorphSeparator)
			for morphPart in morphParts:
				if(morphPart != auxiliaryNeuronsTokenisationSubwordMorphEmpty):
					result = True
		return result

	def tokenSuffixIndicatesAuxiliaryVariation(token):
		if(token.word is None):
			raise RuntimeError("tokenSuffixIndicatesAuxiliaryVariation error: token word is None")
		result = False
		for suffix in auxiliaryNeuronsTokenisationSubwordSuffixList:
			if(token.word.endswith(suffix)):
				if(len(token.word) > len(suffix) + auxiliaryNeuronsTokenisationSubwordSuffixMinimumStemLength):
					result = True
		return result

	def buildAuxiliaryFeatureName(auxiliaryFeatureValue):
		result = auxiliaryNeuronsTokenisationSubwordFeatureNamePrefix + auxiliaryNeuronsTokenisationSubwordFeatureNameDelimiter + auxiliaryFeatureValue
		return result

	def buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, conceptIndex, auxiliaryFeatureWord):
		normalisedConceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, conceptIndex)
		auxiliaryFeatureNamePrefix = auxiliaryNeuronsTokenisationSubwordFeatureNamePrefix + auxiliaryNeuronsTokenisationSubwordFeatureNameDelimiter
		if(not auxiliaryFeatureWord.startswith(auxiliaryFeatureNamePrefix)):
			raise RuntimeError("buildConceptColumnAuxiliaryFeatureName error: auxiliaryFeatureWord missing prefix")
		result = auxiliaryFeatureNamePrefix + str(normalisedConceptIndex) + auxiliaryNeuronsTokenisationSubwordFeatureNameDelimiter + auxiliaryFeatureWord[len(auxiliaryFeatureNamePrefix):]
		return result

	def buildIndexListFromIndexDict(indexDict, mapName):
		result = [None]*len(indexDict)
		for key, index in indexDict.items():
			if(not isinstance(index, int) or isinstance(index, bool)):
				raise RuntimeError("buildIndexListFromIndexDict error: index must be int for " + mapName)
			if(index < 0 or index >= len(indexDict)):
				raise RuntimeError("buildIndexListFromIndexDict error: index out of range for " + mapName)
			if(result[index] is not None):
				raise RuntimeError("buildIndexListFromIndexDict error: duplicate index for " + mapName)
			result[index] = key
		for value in result:
			if(value is None):
				raise RuntimeError("buildIndexListFromIndexDict error: missing index for " + mapName)
		return result

	def loadOrCreateDatabaseAuxiliaryFeatureMaps(loadExistingDatabase):
		auxiliaryFeaturesDict = {}
		auxiliaryFeatureWordsByParentWord = {}
		if(loadExistingDatabase):
			auxiliaryFeaturesDict = GIAANNcmn_databaseNetworkFiles.loadDictFile(auxiliaryNeuronsTokenisationSubwordFeaturesDictFile)
			auxiliaryFeatureWordsByParentWord = GIAANNcmn_databaseNetworkFiles.loadDictFile(auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWordFile)
		auxiliaryFeaturesList = buildIndexListFromIndexDict(auxiliaryFeaturesDict, auxiliaryNeuronsTokenisationSubwordFeaturesDictFileName)
		result = auxiliaryFeaturesDict, auxiliaryFeaturesList, auxiliaryFeatureWordsByParentWord
		return result

	def initialiseDatabaseNetworkAuxiliary(databaseNetworkObject, auxiliaryFeaturesDict, auxiliaryFeaturesList, auxiliaryFeatureWordsByParentWord, auxiliaryLoadExistingDatabase):
		if(not isinstance(auxiliaryLoadExistingDatabase, bool)):
			raise RuntimeError("initialiseDatabaseNetworkAuxiliary error: auxiliaryLoadExistingDatabase must be bool")
		databaseNetworkObject.auxiliaryFeaturesDict = auxiliaryFeaturesDict
		databaseNetworkObject.auxiliaryFeaturesList = auxiliaryFeaturesList
		databaseNetworkObject.auxiliaryFeaturesIndexToWordDict = dict(enumerate(auxiliaryFeaturesList))
		databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord = auxiliaryFeatureWordsByParentWord
		databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureIndicesByParentWord = buildAuxiliaryFeatureIndicesByParentWord(databaseNetworkObject)
		databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordLoadExistingDatabase = auxiliaryLoadExistingDatabase
		databaseNetworkObject.fa = len(auxiliaryFeaturesList)
		return

	def buildAuxiliaryFeatureIndicesByParentWord(databaseNetworkObject):
		result = {}
		for conceptIndex, auxiliaryFeatureWordsByParentWord in databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord.items():
			normalisedConceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, conceptIndex)
			if(not isinstance(auxiliaryFeatureWordsByParentWord, dict)):
				raise RuntimeError("buildAuxiliaryFeatureIndicesByParentWord error: concept parent map must be dict")
			result[normalisedConceptIndex] = {}
			for parentWord, auxiliaryFeatureWords in auxiliaryFeatureWordsByParentWord.items():
				auxiliaryFeatureIndices = []
				for auxiliaryFeatureWord in auxiliaryFeatureWords:
					if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryFeaturesDict):
						raise RuntimeError("buildAuxiliaryFeatureIndicesByParentWord error: missing auxiliary feature word " + auxiliaryFeatureWord)
					auxiliaryFeatureIndices.append(databaseNetworkObject.auxiliaryFeaturesDict[auxiliaryFeatureWord])
				result[normalisedConceptIndex][parentWord] = auxiliaryFeatureIndices
		return result

	def normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, conceptIndex):
		if(isinstance(conceptIndex, bool)):
			raise RuntimeError("normaliseAuxiliaryParentMapConceptIndex error: conceptIndex must not be bool")
		try:
			result = int(conceptIndex)
		except (TypeError, ValueError):
			raise RuntimeError("normaliseAuxiliaryParentMapConceptIndex error: conceptIndex must be int")
		if(result < 0 or result >= databaseNetworkObject.c):
			raise RuntimeError("normaliseAuxiliaryParentMapConceptIndex error: conceptIndex out of range")
		return result

	def saveDatabaseAuxiliaryFeatureMaps(databaseNetworkObject):
		GIAANNcmn_databaseNetworkFiles.saveDictFile(auxiliaryNeuronsTokenisationSubwordFeaturesDictFile, databaseNetworkObject.auxiliaryFeaturesDict)
		GIAANNcmn_databaseNetworkFiles.saveDictFile(auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWordFile, databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord)
		return

	def processAuxiliaryFeatureDetection(databaseNetworkObject, token, isConcept, allowNewFeatures):
		if(not isConcept):
			if(not hasattr(token, "auxiliaryFeatureWords")):
				raise RuntimeError("processAuxiliaryFeatureDetection error: token missing auxiliaryFeatureWords")
		return

	def registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, allowNewFeatures):
		result = None
		if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryFeaturesDict):
			if(not allowNewFeatures):
				raise RuntimeError("registerAuxiliaryFeatureWord error: auxiliary feature word not found while allowNewFeatures is False (" + auxiliaryFeatureWord + ")")
			result = len(databaseNetworkObject.auxiliaryFeaturesDict)
			databaseNetworkObject.auxiliaryFeaturesDict[auxiliaryFeatureWord] = result
			databaseNetworkObject.auxiliaryFeaturesList.append(auxiliaryFeatureWord)
			databaseNetworkObject.auxiliaryFeaturesIndexToWordDict[result] = auxiliaryFeatureWord
			databaseNetworkObject.fa += 1
		else:
			result = databaseNetworkObject.auxiliaryFeaturesDict[auxiliaryFeatureWord]
		return result

	def registerParentAuxiliaryFeatureWord(databaseNetworkObject, conceptIndex, parentWord, auxiliaryFeatureWord):
		normalisedConceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, conceptIndex)
		if(parentWord is None):
			raise RuntimeError("registerParentAuxiliaryFeatureWord error: parentWord is None")
		if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryFeaturesDict):
			raise RuntimeError("registerParentAuxiliaryFeatureWord error: missing auxiliary feature word " + auxiliaryFeatureWord)
		if(normalisedConceptIndex not in databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord):
			databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord[normalisedConceptIndex] = {}
			databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureIndicesByParentWord[normalisedConceptIndex] = {}
		if(parentWord not in databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord[normalisedConceptIndex]):
			databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord[normalisedConceptIndex][parentWord] = []
			databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureIndicesByParentWord[normalisedConceptIndex][parentWord] = []
		if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord[normalisedConceptIndex][parentWord]):
			databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWord[normalisedConceptIndex][parentWord].append(auxiliaryFeatureWord)
			databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureIndicesByParentWord[normalisedConceptIndex][parentWord].append(databaseNetworkObject.auxiliaryFeaturesDict[auxiliaryFeatureWord])
		return

	def getTokenAuxiliaryFeatureIndices(databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures=False, registerParent=False):
		result = []
		if(not isConcept):
			if(not hasattr(token, "auxiliaryFeatureWords")):
				raise RuntimeError("getTokenAuxiliaryFeatureIndices error: token missing auxiliaryFeatureWords")
			for auxiliaryFeatureWord in token.auxiliaryFeatureWords:
				conceptColumnAuxiliaryFeatureWord = buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, conceptIndex, auxiliaryFeatureWord)
				registerAuxiliaryFeatureWord(databaseNetworkObject, conceptColumnAuxiliaryFeatureWord, allowNewFeatures)
				if(registerParent):
					registerParentAuxiliaryFeatureWord(databaseNetworkObject, conceptIndex, token.word, conceptColumnAuxiliaryFeatureWord)
				result.append(databaseNetworkObject.auxiliaryFeaturesDict[conceptColumnAuxiliaryFeatureWord])
		return result

	def initialiseObservedColumnAuxiliaryStorage(observedColumn):
		observedColumn.auxiliaryFeatureConnectionsBySourceFeature = {}
		observedColumn.auxiliaryLoadedSourceFeatureIndices = set()
		observedColumn.auxiliaryTrainPreparedSourceFeatureIndices = set()
		return

	def initialiseObservedColumnProxyAuxiliaryStorage(observedColumnProxy):
		observedColumnProxy.auxiliaryFeatureConnectionsBySourceFeature = {}
		observedColumnProxy.auxiliaryLoadedSourceFeatureIndices = set()
		observedColumnProxy.auxiliaryTrainPreparedSourceFeatureIndices = set()
		return

	def normaliseAuxiliarySourceFeatureIndex(databaseNetworkObject, sourceFeatureIndex):
		result = int(sourceFeatureIndex)
		if(result < 0 or result >= databaseNetworkObject.fa):
			raise RuntimeError("normaliseAuxiliarySourceFeatureIndex error: source feature index out of range")
		return result

	def normaliseAuxiliarySourceFeatureIndices(databaseNetworkObject, sourceFeatureIndices):
		result = []
		seen = set()
		if(sourceFeatureIndices is not None):
			if(pt.is_tensor(sourceFeatureIndices)):
				rawSourceFeatureIndices = sourceFeatureIndices.detach().view(-1).cpu().tolist()
			else:
				rawSourceFeatureIndices = list(sourceFeatureIndices)
			for sourceFeatureIndex in rawSourceFeatureIndices:
				normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(databaseNetworkObject, sourceFeatureIndex)
				if(normalisedSourceFeatureIndex not in seen):
					result.append(normalisedSourceFeatureIndex)
					seen.add(normalisedSourceFeatureIndex)
		result.sort()
		return result

	def initialiseAuxiliaryFeatureConnections(databaseNetworkObject, targetDevice):
		indices = pt.empty((5, 0), dtype=pt.long, device=targetDevice)
		values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f), dtype=arrayType, device=targetDevice)
		return result

	def getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex):
		result = os.path.join(GIAANNcmn_databaseNetworkFiles.getObservedColumnFolder(conceptIndex), auxiliaryNeuronsTokenisationSubwordConnectionsFolderName)
		return result

	def getObservedColumnAuxiliarySourceFeatureConnectionsFileBaseName(sourceFeatureIndex):
		result = auxiliaryNeuronsTokenisationSubwordSourceFeatureConnectionsFileNamePrefix + str(int(sourceFeatureIndex))
		return result

	def listObservedColumnAuxiliarySourceFeatureIndices(databaseNetworkObject, conceptIndex):
		result = []
		connectionsFolder = getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex)
		if(os.path.isdir(connectionsFolder)):
			for fileName in os.listdir(connectionsFolder):
				if(fileName.startswith(auxiliaryNeuronsTokenisationSubwordSourceFeatureConnectionsFileNamePrefix) and fileName.endswith(pytorchTensorFileExtension)):
					filePath = os.path.join(connectionsFolder, fileName)
					if(databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordLoadExistingDatabase or filePath in auxiliaryNeuronsTokenisationSubwordSavedSourceTensorPaths):
						sourceFeatureIndexString = fileName[len(auxiliaryNeuronsTokenisationSubwordSourceFeatureConnectionsFileNamePrefix):-len(pytorchTensorFileExtension)]
						result.append(int(sourceFeatureIndexString))
		result.sort()
		return result

	def loadObservedColumnAuxiliaryFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, targetDevice, ensureCurrentSizeOnLoad=False):
		connectionsFolder = getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex)
		fileBaseName = getObservedColumnAuxiliarySourceFeatureConnectionsFileBaseName(sourceFeatureIndex)
		tensorName = "observedColumn.auxiliaryFeatureConnectionsBySourceFeature[" + str(conceptIndex) + "][" + str(sourceFeatureIndex) + "]"
		tensor = GIAANNcmn_databaseNetworkFiles.adjustPropertyDimensions(databaseNetworkObject.inferenceMode, GIAANNcmn_databaseNetworkFiles.loadTensor(connectionsFolder, fileBaseName, targetDevice=targetDevice), tensorName)
		tensor = GIAANNcmn_databaseNetworkFiles.adjustBranchDimensions(tensor, tensorName, expectedRank=5)
		if(ensureCurrentSizeOnLoad):
			tensor = GIAANNcmn_databaseNetworkFiles.ensureFeatureConnectionsSourceTensorCurrentSize(tensor, databaseNetworkObject.c, databaseNetworkObject.f, tensorName)
		return tensor

	def saveObservedColumnAuxiliaryFeatureConnectionsTensor(conceptIndex, sourceFeatureIndex, tensor):
		connectionsFolder = getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex)
		fileBaseName = getObservedColumnAuxiliarySourceFeatureConnectionsFileBaseName(sourceFeatureIndex)
		filePath = os.path.join(connectionsFolder, fileBaseName + pytorchTensorFileExtension)
		if(tensor is None):
			raise RuntimeError("saveObservedColumnAuxiliaryFeatureConnectionsTensor error: tensor is None")
		os.makedirs(connectionsFolder, exist_ok=True)
		if(tensor.is_sparse):
			tensor = tensor.coalesce()
			tensorNNZ = tensor._nnz()
		else:
			tensorNNZ = int(pt.count_nonzero(tensor).item())
		if(tensorNNZ > 0):
			GIAANNcmn_databaseNetworkFiles.saveTensor(tensor, connectionsFolder, fileBaseName)
			auxiliaryNeuronsTokenisationSubwordSavedSourceTensorPaths.add(filePath)
		else:
			if(GIAANNcmn_databaseNetworkFiles.pathExists(filePath)):
				os.remove(filePath)
			auxiliaryNeuronsTokenisationSubwordSavedSourceTensorPaths.discard(filePath)
		return

	def getObservedColumnAuxiliaryFeatureConnectionsTargetSize(observedColumn):
		result = (observedColumn.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, observedColumn.databaseNetworkObject.c, observedColumn.databaseNetworkObject.f)
		return result

	def ensureObservedColumnAuxiliaryStorage(observedColumn):
		if(not hasattr(observedColumn, "auxiliaryFeatureConnectionsBySourceFeature")):
			initialiseObservedColumnAuxiliaryStorage(observedColumn)
		return

	def getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=None, createMissing=False, ensureCurrentSizeOnLoad=False):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
		resolvedTargetDevice = targetDevice if targetDevice is not None else deviceSparse
		result = observedColumn.auxiliaryFeatureConnectionsBySourceFeature.get(normalisedSourceFeatureIndex)
		if(result is None):
			if(hasattr(observedColumn, "sourceObservedColumn")):
				sourceTensor = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn.sourceObservedColumn, normalisedSourceFeatureIndex, observedColumn.sourceObservedColumn.getDefaultConnectionTargetDevice(), createMissing, ensureCurrentSizeOnLoad)
				result = sourceTensor.to(resolvedTargetDevice)
			elif(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and observedColumn.databaseNetworkObject.observedColumnsRAMLoaded):
				result = initialiseAuxiliaryFeatureConnections(observedColumn.databaseNetworkObject, resolvedTargetDevice)
			else:
				storedSourceFeatureIndices = listObservedColumnAuxiliarySourceFeatureIndices(observedColumn.databaseNetworkObject, observedColumn.conceptIndex)
				if(normalisedSourceFeatureIndex in storedSourceFeatureIndices):
					result = loadObservedColumnAuxiliaryFeatureConnectionsTensor(observedColumn.databaseNetworkObject, observedColumn.conceptIndex, normalisedSourceFeatureIndex, resolvedTargetDevice, ensureCurrentSizeOnLoad=ensureCurrentSizeOnLoad)
				else:
					result = initialiseAuxiliaryFeatureConnections(observedColumn.databaseNetworkObject, resolvedTargetDevice)
			observedColumn.auxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		elif(result.device != resolvedTargetDevice):
			result = result.to(resolvedTargetDevice)
			observedColumn.auxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		if(ensureCurrentSizeOnLoad):
			ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, normalisedSourceFeatureIndex)
			result = observedColumn.auxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex]
			if(result.device != resolvedTargetDevice):
				result = result.to(resolvedTargetDevice)
				observedColumn.auxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		observedColumn.auxiliaryLoadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return result

	def setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, tensor):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
		expectedSize = getObservedColumnAuxiliaryFeatureConnectionsTargetSize(observedColumn)
		if(tensor is None):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature error: tensor is None")
		if(tensor.layout != pt.sparse_coo):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature error: tensor must be sparse COO")
		if(tensor.dim() != 5):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature error: tensor rank must be 5")
		if(tuple(tensor.size()) != tuple(expectedSize)):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature error: tensor size mismatch")
		if(not tensor.is_coalesced()):
			tensor = tensor.coalesce()
		observedColumn.auxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = tensor
		observedColumn.auxiliaryLoadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return

	def prepareObservedColumnAuxiliaryFeatureConnectionsTrain(observedColumn, requiredSourceFeatureIndices, targetDevice):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		resolvedSourceFeatureIndices = normaliseAuxiliarySourceFeatureIndices(observedColumn.databaseNetworkObject, requiredSourceFeatureIndices)
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=targetDevice, createMissing=False)
			ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, sourceFeatureIndex)
		observedColumn.auxiliaryTrainPreparedSourceFeatureIndices = set(resolvedSourceFeatureIndices)
		return

	def ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, sourceFeatureIndex):
		normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
		if(normalisedSourceFeatureIndex not in observedColumn.auxiliaryFeatureConnectionsBySourceFeature):
			raise RuntimeError("ensureObservedColumnAuxiliaryFeatureConnectionSize error: missing loaded source feature tensor")
		sourceTensor = observedColumn.auxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex]
		expectedSize = getObservedColumnAuxiliaryFeatureConnectionsTargetSize(observedColumn)
		if(tuple(sourceTensor.size()) != tuple(expectedSize)):
			sourceTensor = GIAANNcmn_databaseNetworkFiles.expandSparseTensorSize(sourceTensor, expectedSize, "ensureObservedColumnAuxiliaryFeatureConnectionSize")
			observedColumn.auxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = sourceTensor
		return

	def saveObservedColumnAuxiliaryFeatureConnectionsToDisk(observedColumn, saveAllSourceFeatures):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		if(saveAllSourceFeatures):
			sourceFeatureIndicesToSave = sorted(observedColumn.auxiliaryLoadedSourceFeatureIndices)
		else:
			sourceFeatureIndicesToSave = sorted(observedColumn.auxiliaryTrainPreparedSourceFeatureIndices)
		for sourceFeatureIndex in sourceFeatureIndicesToSave:
			if(sourceFeatureIndex not in observedColumn.auxiliaryFeatureConnectionsBySourceFeature):
				raise RuntimeError("saveObservedColumnAuxiliaryFeatureConnectionsToDisk error: missing loaded source feature tensor")
			saveObservedColumnAuxiliaryFeatureConnectionsTensor(observedColumn.conceptIndex, sourceFeatureIndex, observedColumn.auxiliaryFeatureConnectionsBySourceFeature[sourceFeatureIndex])
		return

	def loadObservedColumnAuxiliaryConnectionsFromDisk(observedColumn, targetDevice=None, loadAllSourceFeatures=False, resizeFeatureTensorsToCurrentSize=False):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		if(loadAllSourceFeatures):
			sourceFeatureIndices = listObservedColumnAuxiliarySourceFeatureIndices(observedColumn.databaseNetworkObject, observedColumn.conceptIndex)
			loadTargetDevice = targetDevice if targetDevice is not None else deviceDatabase
			for sourceFeatureIndex in sourceFeatureIndices:
				getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=loadTargetDevice, createMissing=False, ensureCurrentSizeOnLoad=resizeFeatureTensorsToCurrentSize)
		return

	def ensureRAMdatabaseAuxiliaryFeatureTensorSizes(observedColumn):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		for sourceFeatureIndex in sorted(observedColumn.auxiliaryFeatureConnectionsBySourceFeature.keys()):
			ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, sourceFeatureIndex)
		return

	def moveObservedColumnAuxiliaryConnectionsToDatabaseAfterTrain(observedColumn):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		for sourceFeatureIndex in sorted(observedColumn.auxiliaryTrainPreparedSourceFeatureIndices):
			sourceTensor = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=deviceDatabase, createMissing=False)
			setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, sourceTensor)
		observedColumn.auxiliaryTrainPreparedSourceFeatureIndices.clear()
		return

	def getRequiredAuxiliaryFeatureIndicesByObservedColumn(sequenceObservedColumns, allowNewFeatures):
		if(not isinstance(allowNewFeatures, bool)):
			raise RuntimeError("getRequiredAuxiliaryFeatureIndicesByObservedColumn error: allowNewFeatures must be bool")
		result = {}
		sequenceObservedColumns.ensureTokenConceptColumnIndexList()
		for observedColumn in sequenceObservedColumns.observedColumnsDict.values():
			result[int(observedColumn.conceptIndex)] = set()
		for tokenIndex, token in enumerate(sequenceObservedColumns.tokens):
			isConcept = tokenIndex in sequenceObservedColumns.columnsIndexSequenceWordIndexDict
			if(not isConcept):
				conceptIndex = sequenceObservedColumns.tokenConceptColumnIndexList[tokenIndex]
				if(conceptIndex is None):
					raise RuntimeError("getRequiredAuxiliaryFeatureIndicesByObservedColumn error: unassigned token")
				if(int(conceptIndex) not in result):
					raise RuntimeError("getRequiredAuxiliaryFeatureIndicesByObservedColumn error: missing observed column")
				if(not hasattr(token, "auxiliaryFeatureWords")):
					raise RuntimeError("getRequiredAuxiliaryFeatureIndicesByObservedColumn error: token missing auxiliaryFeatureWords")
				for auxiliaryFeatureIndex in getTokenAuxiliaryFeatureIndices(sequenceObservedColumns.databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures, registerParent=True):
					result[int(conceptIndex)].add(int(auxiliaryFeatureIndex))
		for conceptIndex, auxiliaryFeatureIndices in result.items():
			result[conceptIndex] = sorted(auxiliaryFeatureIndices)
		sequenceObservedColumns.requiredAuxiliarySourceFeatureIndicesByObservedColumn = result
		return result

	def prepareObservedColumnsForTrainSequenceAuxiliary(sequenceObservedColumns, observedColumnsDict, allowNewFeatures):
		requiredAuxiliaryFeatureIndicesByObservedColumn = getRequiredAuxiliaryFeatureIndicesByObservedColumn(sequenceObservedColumns, allowNewFeatures)
		for observedColumn in observedColumnsDict.values():
			conceptIndex = int(observedColumn.conceptIndex)
			if(conceptIndex not in requiredAuxiliaryFeatureIndicesByObservedColumn):
				raise RuntimeError("prepareObservedColumnsForTrainSequenceAuxiliary error: missing required auxiliary source features")
			prepareObservedColumnAuxiliaryFeatureConnectionsTrain(observedColumn, requiredAuxiliaryFeatureIndicesByObservedColumn[conceptIndex], deviceSparse)
		return

	def buildTokenSequenceConceptIndexList(tokens, conceptIndices, startIndices, endIndices):
		result = [None]*len(tokens)
		conceptIndicesList = conceptIndices.tolist()
		startList = startIndices.tolist()
		endList = endIndices.tolist()
		for sequenceConceptIndex, conceptWordIndex in enumerate(conceptIndicesList):
			startIndex = max(0, int(startList[sequenceConceptIndex]))
			endIndex = min(len(tokens), int(endList[sequenceConceptIndex]))
			for tokenIndex in range(startIndex, endIndex):
				result[tokenIndex] = sequenceConceptIndex
		return result

	def trainAuxiliaryFeatureConnections(sequenceObservedColumns, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, conceptIndices, startIndices, endIndices):
		if(arrayIndexPropertiesPos or trainConnectionStrengthPOSdependence):
			raise RuntimeError("trainAuxiliaryFeatureConnections error: auxiliary POS connection properties are not implemented")
		sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder = buildAuxiliarySourceOccurrenceTensors(sequenceObservedColumns, conceptIndices, startIndices, endIndices, featureNeuronsActive.device)
		connectionIndices, connectionValues = buildAuxiliaryConnectionIndicesAndValues(sequenceObservedColumns, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder)
		if(connectionIndices.numel() > 0):
			applyAuxiliaryConnectionPropertyUpdates(sequenceObservedColumns, connectionIndices, connectionValues, sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesStrengthIndex)
			if(arrayIndexPropertiesPermanence):
				permanenceValues = pt.full((connectionValues.shape[0],), z1, dtype=arrayType, device=connectionValues.device)
				applyAuxiliaryConnectionPropertyUpdates(sequenceObservedColumns, connectionIndices, permanenceValues, sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesPermanenceIndex)
		return

	def buildAuxiliarySourceOccurrenceTensors(sequenceObservedColumns, conceptIndices, startIndices, endIndices, targetDevice):
		sourceConceptIndexList = []
		sourceAuxiliaryFeatureIndexList = []
		sourceWordOrderList = []
		tokenSequenceConceptIndexList = buildTokenSequenceConceptIndexList(sequenceObservedColumns.tokens, conceptIndices, startIndices, endIndices)
		for tokenIndex, token in enumerate(sequenceObservedColumns.tokens):
			isConcept = tokenIndex in sequenceObservedColumns.columnsIndexSequenceWordIndexDict
			if(not isConcept):
				sequenceConceptIndex = tokenSequenceConceptIndexList[tokenIndex]
				if(sequenceConceptIndex is None):
					raise RuntimeError("buildAuxiliarySourceOccurrenceTensors error: token has no sequence concept index")
				conceptIndex = sequenceObservedColumns.tokenConceptColumnIndexList[tokenIndex]
				if(conceptIndex is None):
					raise RuntimeError("buildAuxiliarySourceOccurrenceTensors error: token has no concept index")
				for auxiliaryFeatureIndex in getTokenAuxiliaryFeatureIndices(sequenceObservedColumns.databaseNetworkObject, token, isConcept, conceptIndex):
					sourceConceptIndexList.append(sequenceConceptIndex)
					sourceAuxiliaryFeatureIndexList.append(auxiliaryFeatureIndex)
					sourceWordOrderList.append(tokenIndex)
		sourceConceptIndices = pt.tensor(sourceConceptIndexList, dtype=pt.long, device=targetDevice)
		sourceAuxiliaryFeatureIndices = pt.tensor(sourceAuxiliaryFeatureIndexList, dtype=pt.long, device=targetDevice)
		sourceWordOrder = pt.tensor(sourceWordOrderList, dtype=pt.long, device=targetDevice)
		result = sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder
		return result

	def buildAuxiliaryConnectionIndicesAndValues(sequenceObservedColumns, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder):
		connectionDevice = featureNeuronsActive.device
		indicesList = []
		valuesList = []
		if(sourceConceptIndices.numel() > 0):
			targetActive = featureNeuronsActive.amax(dim=1)
			targetIndices = pt.nonzero(targetActive > 0, as_tuple=False)
			if(targetIndices.numel() > 0):
				sourceCount = sourceConceptIndices.shape[0]
				targetCount = targetIndices.shape[0]
				branchIndices = targetIndices[:, 0].repeat(sourceCount)
				targetConceptIndices = targetIndices[:, 1].repeat(sourceCount)
				targetFeatureIndices = targetIndices[:, 2].repeat(sourceCount)
				sourceConceptIndicesPair = sourceConceptIndices.repeat_interleave(targetCount)
				sourceAuxiliaryFeatureIndicesPair = sourceAuxiliaryFeatureIndices.repeat_interleave(targetCount)
				sourceWordOrderPair = sourceWordOrder.repeat_interleave(targetCount)
				targetWordOrder = featureNeuronsWordOrder[targetConceptIndices, targetFeatureIndices]
				connectionMask = createAuxiliaryFeatureWordOrderConnectionMask(sourceWordOrderPair, targetWordOrder, sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex)
				sourceColumnWordOrder = columnsWordOrder[sourceConceptIndicesPair]
				targetColumnWordOrder = columnsWordOrder[targetConceptIndices]
				if(debugConnectColumnsToNextColumnsInSequenceOnly):
					connectionMask = connectionMask & pt.logical_and(targetColumnWordOrder >= sourceColumnWordOrder, targetColumnWordOrder <= sourceColumnWordOrder+1)
				else:
					connectionMask = connectionMask & (targetColumnWordOrder >= sourceColumnWordOrder)
				if(connectionMask.any()):
					branchIndices = branchIndices[connectionMask]
					sourceConceptIndicesPair = sourceConceptIndicesPair[connectionMask]
					sourceAuxiliaryFeatureIndicesPair = sourceAuxiliaryFeatureIndicesPair[connectionMask]
					targetConceptIndices = targetConceptIndices[connectionMask]
					targetFeatureIndices = targetFeatureIndices[connectionMask]
					sourceWordOrderPair = sourceWordOrderPair[connectionMask]
					targetWordOrder = targetWordOrder[connectionMask]
					baseValues = pt.ones((branchIndices.shape[0],), dtype=arrayType, device=connectionDevice)
					if(trainConnectionStrengthNormaliseWrtContextLength):
						connectionDistances = pt.abs(targetWordOrder - sourceWordOrderPair).to(baseValues.dtype)
						baseValues = baseValues * (auxiliaryNeuronsTokenisationSubwordConnectionProximityMultiplier/(connectionDistances + 1))
					appendAuxiliaryConnectionSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndicesPair, sourceAuxiliaryFeatureIndicesPair, targetConceptIndices, targetFeatureIndices, sourceWordOrderPair, targetWordOrder, baseValues)
		if(len(indicesList) > 0):
			combinedIndices = pt.cat(indicesList, dim=1)
			combinedValues = pt.cat(valuesList, dim=0)
			sparseSize = (numberOfDendriticBranches, arrayNumberOfSegments, sequenceObservedColumns.cs, sequenceObservedColumns.databaseNetworkObject.fa, sequenceObservedColumns.cs, sequenceObservedColumns.fs)
			connectionSparse = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=sparseSize, dtype=arrayType, device=connectionDevice).coalesce()
			combinedIndices = connectionSparse.indices()
			combinedValues = connectionSparse.values()
		else:
			combinedIndices = pt.empty((6, 0), dtype=pt.long, device=connectionDevice)
			combinedValues = pt.empty((0,), dtype=arrayType, device=connectionDevice)
		result = combinedIndices, combinedValues
		return result

	def createAuxiliaryFeatureWordOrderConnectionMask(sourceWordOrder, targetWordOrder, trainConnectionsIncludeSameTimeIndex):
		if(not isinstance(trainConnectionsIncludeSameTimeIndex, bool)):
			raise RuntimeError("createAuxiliaryFeatureWordOrderConnectionMask error: trainConnectionsIncludeSameTimeIndex must be a bool")
		if(debugConnectNodesToNextNodesInSequenceOnly):
			wordOrderUpperBound = sourceWordOrder + 1
			if(trainConnectionsIncludeSameTimeIndex):
				result = pt.logical_and(targetWordOrder >= sourceWordOrder, targetWordOrder <= wordOrderUpperBound)
			else:
				result = pt.logical_and(targetWordOrder > sourceWordOrder, targetWordOrder <= wordOrderUpperBound)
		else:
			if(trainConnectionsIncludeSameTimeIndex):
				result = targetWordOrder >= sourceWordOrder
			else:
				result = targetWordOrder > sourceWordOrder
		return result

	def appendAuxiliaryConnectionSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, baseValues):
		if(sourceConceptIndices.numel() > 0):
			if(useSANIcolumns):
				conceptDistances = pt.abs(targetConceptIndices - sourceConceptIndices)
				connectionsSegmentIndex = arrayNumberOfSegments - conceptDistances - 1
				connectionsSegmentIndex = pt.clamp(connectionsSegmentIndex, min=0, max=arrayNumberOfSegments-1).long()
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
			elif(useSANIfeatures):
				relativeDistance = targetWordOrder - sourceWordOrder
				if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
					relativeDistance = pt.clamp(relativeDistance, min=1)
					connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
					connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
					appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
				else:
					relativeDistance = pt.clamp(relativeDistance, min=1)
					validDistanceMask = relativeDistance <= arrayNumberOfSegments
					if(validDistanceMask.any()):
						connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
						connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
						appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices[validDistanceMask], connectionsSegmentIndex[validDistanceMask], sourceConceptIndices[validDistanceMask], sourceAuxiliaryFeatureIndices[validDistanceMask], targetConceptIndices[validDistanceMask], targetFeatureIndices[validDistanceMask], baseValues[validDistanceMask])
			elif(useSANIfeaturesAndColumns):
				appendAuxiliaryFeatureAndColumnSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, baseValues)
			else:
				connectionsSegmentIndex = pt.zeros((branchIndices.shape[0],), dtype=pt.long, device=branchIndices.device)
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
		return

	def appendAuxiliaryFeatureAndColumnSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, baseValues):
		relativeDistance = targetWordOrder - sourceWordOrder
		featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
		if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
			relativeDistanceFeature = pt.clamp(relativeDistance, min=1, max=arrayNumberOfSegmentsFeatureDistance)
			featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistanceFeature
			featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
			appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, featureSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
		else:
			relativeDistanceFeature = pt.clamp(relativeDistance, min=1)
			validFeatureDistanceMask = relativeDistanceFeature <= arrayNumberOfSegmentsFeatureDistance
			if(validFeatureDistanceMask.any()):
				featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistanceFeature
				featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices[validFeatureDistanceMask], featureSegmentIndex[validFeatureDistanceMask], sourceConceptIndices[validFeatureDistanceMask], sourceAuxiliaryFeatureIndices[validFeatureDistanceMask], targetConceptIndices[validFeatureDistanceMask], targetFeatureIndices[validFeatureDistanceMask], baseValues[validFeatureDistanceMask])
		if(arrayNumberOfSegmentsColumnDistance > 0):
			conceptDistances = pt.abs(targetConceptIndices - sourceConceptIndices)
			if(useSANIfeaturesAndColumnsInternal):
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptDistances - 1
			else:
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptDistances
			columnSegmentIndex = columnSegmentIndex.clamp(min=0, max=arrayNumberOfSegmentsColumnDistance-1).long()
			validColumnMask = pt.ones((branchIndices.shape[0],), dtype=pt.bool, device=branchIndices.device)
			if(not useSANIfeaturesAndColumnsInternal):
				validColumnMask = conceptDistances > 0
			if(validColumnMask.any()):
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices[validColumnMask], columnSegmentIndex[validColumnMask], sourceConceptIndices[validColumnMask], sourceAuxiliaryFeatureIndices[validColumnMask], targetConceptIndices[validColumnMask], targetFeatureIndices[validColumnMask], baseValues[validColumnMask])
		return

	def appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, segmentIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, values):
		indicesList.append(pt.stack((branchIndices, segmentIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices), dim=0))
		valuesList.append(values)
		return

	def applyAuxiliaryConnectionPropertyUpdates(sequenceObservedColumns, connectionIndices, connectionValues, propertyIndex):
		if(connectionIndices.numel() > 0):
			connectionDevice = connectionIndices.device
			databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
			conceptIndicesTensor = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(connectionDevice)
			featureIndicesInObserved = sequenceObservedColumns.featureIndicesInObservedTensor.to(connectionDevice)
			sourceConceptIndex = conceptIndicesTensor[connectionIndices[2]]
			sourceAuxiliaryFeatureIndex = connectionIndices[3]
			targetConceptIndex = conceptIndicesTensor[connectionIndices[4]]
			targetFeatureIndex = featureIndicesInObserved[connectionIndices[5]]
			sourceCombinedKeys = sourceConceptIndex * databaseNetworkObject.fa + sourceAuxiliaryFeatureIndex
			sortedSourceCombinedKeys, sortOrder = pt.sort(sourceCombinedKeys)
			sortedBranch = connectionIndices[0].index_select(0, sortOrder)
			sortedSegment = connectionIndices[1].index_select(0, sortOrder)
			sortedTargetConceptIndex = targetConceptIndex.index_select(0, sortOrder)
			sortedTargetFeatureIndex = targetFeatureIndex.index_select(0, sortOrder)
			sortedValues = connectionValues.index_select(0, sortOrder)
			uniqueSourceCombinedKeys, counts = pt.unique_consecutive(sortedSourceCombinedKeys, return_counts=True)
			starts = pt.cumsum(counts, 0) - counts
			if(trainSequenceObservedColumnsMatchSequenceWords):
				sequenceObservedColumnsDict = sequenceObservedColumns.sequenceObservedColumnsDict
			else:
				sequenceObservedColumnsDict = sequenceObservedColumns.observedColumnsDict2
			observedColumnsByConceptIndex = sequenceObservedColumns.getObservedColumnsByConceptIndex(sequenceObservedColumnsDict)
			targetSize = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
			for sourceCombinedKey, start, count in zip(uniqueSourceCombinedKeys.tolist(), starts.tolist(), counts.tolist()):
				end = start + count
				sourceConceptIndexValue = int(sourceCombinedKey // databaseNetworkObject.fa)
				sourceAuxiliaryFeatureIndexValue = int(sourceCombinedKey % databaseNetworkObject.fa)
				if(sourceConceptIndexValue not in observedColumnsByConceptIndex):
					raise RuntimeError("applyAuxiliaryConnectionPropertyUpdates error: missing observed column")
				observedColumn = observedColumnsByConceptIndex[sourceConceptIndexValue]
				if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
					if(sourceAuxiliaryFeatureIndexValue not in observedColumn.auxiliaryTrainPreparedSourceFeatureIndices):
						raise RuntimeError("applyAuxiliaryConnectionPropertyUpdates error: source auxiliary feature was not prepared")
				propertyRow = pt.full((count,), propertyIndex, dtype=pt.long, device=connectionDevice)
				updateIndices = pt.stack((propertyRow, sortedBranch[start:end], sortedSegment[start:end], sortedTargetConceptIndex[start:end], sortedTargetFeatureIndex[start:end]), dim=0)
				updateSparse = pt.sparse_coo_tensor(updateIndices, sortedValues[start:end], size=targetSize, dtype=arrayType, device=connectionDevice)
				targetSparse = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceAuxiliaryFeatureIndexValue, targetDevice=connectionDevice, createMissing=False)
				targetSparse = addSparseUpdateNonNegative(targetSparse, updateSparse)
				setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceAuxiliaryFeatureIndexValue, targetSparse)
		return

	def addSparseUpdateNonNegative(targetSparse, updateSparse):
		result = (targetSparse.coalesce() + updateSparse.coalesce()).coalesce()
		if(result._nnz() > 0):
			resultValues = result.values()
			resultValues.clamp_(min=0)
		return result

	def getAuxiliaryFeatureIndicesForParentFeature(databaseNetworkObject, observedColumn, sourceFeatureIndex):
		result = []
		if(observedColumn is None):
			raise RuntimeError("getAuxiliaryFeatureIndicesForParentFeature error: observedColumn is None")
		if(not hasattr(observedColumn, "conceptIndex")):
			raise RuntimeError("getAuxiliaryFeatureIndicesForParentFeature error: observedColumn missing conceptIndex")
		if(sourceFeatureIndex != featureIndexPrimeConceptNeuron):
			if(sourceFeatureIndex < 0 or sourceFeatureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
				raise RuntimeError("getAuxiliaryFeatureIndicesForParentFeature error: sourceFeatureIndex out of range")
			conceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, observedColumn.conceptIndex)
			parentWord = databaseNetworkObject.conceptFeaturesList[sourceFeatureIndex]
			if(conceptIndex in databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureIndicesByParentWord):
				if(parentWord in databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureIndicesByParentWord[conceptIndex]):
					result = databaseNetworkObject.auxiliaryNeuronsTokenisationSubwordFeatureIndicesByParentWord[conceptIndex][parentWord]
		return result

	def processAuxiliaryFeaturePredictionActivations(databaseNetworkObject, observedColumn, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sourceColumnIndex, sourceFeatureIndex, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None):
		import GIAANNcmn_predictionActivate
		globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
		globalFeatureConnectionsActivationResult = globalFeatureConnectionsActivation
		globalFeatureNeuronsTimeResult = globalFeatureNeuronsTime
		auxiliaryFeatureIndices = getAuxiliaryFeatureIndicesForParentFeature(databaseNetworkObject, observedColumn, int(sourceFeatureIndex))
		for auxiliaryFeatureIndex in auxiliaryFeatureIndices:
			connectionDevice = globalFeatureNeuronsActivationResult.device
			auxiliaryFeatureConnections = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, auxiliaryFeatureIndex, targetDevice=connectionDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
			globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult = GIAANNcmn_predictionActivate.processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, auxiliaryFeatureConnections, sourceColumnIndex, sourceFeatureIndex, globalFeatureNeuronsTimeResult, sequenceWordIndex, sequenceColumnIndex)
		result = globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult
		return result

	def getConnectedColumnsForAuxiliaryFeatures(observedColumn, parentFeatureIndex, includeFeatureDetails=False):
		targetColumnsList = []
		columnFeatureMap = {}
		databaseNetworkObject = observedColumn.databaseNetworkObject
		auxiliaryFeatureIndices = getAuxiliaryFeatureIndicesForParentFeature(databaseNetworkObject, observedColumn, int(parentFeatureIndex))
		for auxiliaryFeatureIndex in auxiliaryFeatureIndices:
			auxiliaryFeatureConnections = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, auxiliaryFeatureIndex, targetDevice=deviceSparse, createMissing=False)
			featureConnectionsStrength = auxiliaryFeatureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
			featureConnectionsStrength = featureConnectionsStrength.coalesce()
			if(featureConnectionsStrength._nnz() > 0):
				targetColumnIndices = featureConnectionsStrength.indices()
				if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
					lastSegmentMask = targetColumnIndices[1] == arrayIndexSegmentLast
					targetColumnIndices = targetColumnIndices[:, lastSegmentMask]
				targetColumns = targetColumnIndices[2].unique()
				targetColumnsList.extend(targetColumns.cpu().tolist())
				if(includeFeatureDetails):
					for columnValue, featureValue in zip(targetColumnIndices[2].tolist(), targetColumnIndices[3].cpu().tolist()):
						columnFeatureMap.setdefault(columnValue, set()).add(featureValue)
		targetColumnsList = sorted(set(targetColumnsList))
		result = targetColumnsList, columnFeatureMap
		return result
