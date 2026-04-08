"""GIAANNproto_sequenceObservedColumns.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto sequence Observed Columns

"""

import torch as pt
import time

from GIAANNproto_globalDefs import *
import GIAANNproto_debug
import GIAANNproto_sparseTensors
import GIAANNproto_sequenceConcepts
if(optimisationUseCUDAObservedColumnUpdateKernel):
	import GIAANNproto_cudaObservedColumnUpdate

# Define the SequenceObservedColumns class
class SequenceObservedColumns:
	"""
	Contains sequence observed columns object arrays which stack a feature subset of the observed columns object arrays for the current sequence.
	"""
	def __init__(self, databaseNetworkObject, tokens, observedColumnsDict, observedColumnsSequenceWordIndexDict, inferenceMode):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.databaseNetworkObject = databaseNetworkObject
		self.observedColumnsDict = observedColumnsDict	# key: lemma, value: ObservedColumn
		self.observedColumnsSequenceWordIndexDict = observedColumnsSequenceWordIndexDict	# key: sequence word index, value: ObservedColumn
		self.noDelimiterDetectedBetweenConceptTokens = False
		self.requiredSourceFeatureIndicesByObservedColumn = None

		if(trainSequenceObservedColumnsMatchSequenceWords):
			self.cs = len(observedColumnsSequenceWordIndexDict)

			self.columnsIndexSequenceWordIndexDict = {}	#key: sequence word index, value: conceptIndex
			self.sequenceObservedColumnsDict = {}	#key: sequenceConceptIndex, value: observedColumn 
			self.conceptIndicesInObservedList = []	#value: concept index
			for sequenceConceptIndex, (sequenceWordIndex, observedColumn) in enumerate(self.observedColumnsSequenceWordIndexDict.items()):
				self.columnsIndexSequenceWordIndexDict[sequenceWordIndex] = observedColumn.conceptIndex
				self.sequenceObservedColumnsDict[sequenceConceptIndex] = observedColumn
				self.conceptIndicesInObservedList.append(observedColumn.conceptIndex)
			self.conceptIndicesInSequenceObservedTensor = pt.tensor(self.conceptIndicesInObservedList, dtype=pt.long)
		else:
			self.cs = len(observedColumnsDict) 

			self.columnsIndexSequenceWordIndexDict = {}	#key: sequence word index, value: conceptIndex
			for idx, (sequenceWordIndex, observedColumn) in enumerate(observedColumnsSequenceWordIndexDict.items()):
				self.columnsIndexSequenceWordIndexDict[sequenceWordIndex] = observedColumn.conceptIndex

			# Map from concept names to indices in sequence arrays
			self.conceptIndicesInObservedList = []
			self.conceptNameToIndex = {}	# key: lemma, value: sequenceConceptIndex
			self.indexToConceptName = {}	# key: sequenceConceptIndex, value: lemma
			self.observedColumnsDict2 = {}	# key: sequenceConceptIndex, value: ObservedColumn
			for idx, (lemma, observedColumn) in enumerate(observedColumnsDict.items()):
				self.conceptIndicesInObservedList.append(observedColumn.conceptIndex)
				self.conceptNameToIndex[lemma] = idx
				self.indexToConceptName[idx] = lemma
				self.observedColumnsDict2[idx] = observedColumn
			self.conceptIndicesInSequenceObservedTensor = pt.tensor(self.conceptIndicesInObservedList, dtype=pt.long)
				
		self.featureNeuronChanges = [None]*self.cs
			
		# Collect all feature words from observed columns
		self.tokens = tokens
		skipObservedColumnArrays = inferenceMode and (inferenceOnlyRetainPredictedTargetObservedColumn or (not drawNetworkDuringInference and not drawSequenceObservedColumns and not drawAllColumns))
		if(not skipObservedColumnArrays):
			#identify feature indices from complete ObservedColumns.featureNeurons or globalFeatureNeurons feature lists currently stored in SequenceObservedColumns.featureNeurons	#required for useInference
			observedColumn = list(observedColumnsDict.values())[0]	#all features (including words) are identical per observed column
			self.featureWords, self.featureIndicesInObservedTensor, self.fIdxTensor = self.identifyObservedColumnFeatureWords(tokens, observedColumn)

			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				self.fs = self.featureIndicesInObservedTensor.shape[0]
			else:
				self.fs = len(self.featureWords)
			self.featureWordToIndex = {}
			self.indexToFeatureWord = {}
			for idx, featureWord in enumerate(self.featureWords):
				self.featureWordToIndex[featureWord] = idx
				self.indexToFeatureWord[idx] = featureWord

			self.columnStartIndicesTensor = None
			self.columnEndIndicesTensor = None
			self.columnFeatureLocalIndices = None
			self.computeColumnLocalFeatureMaps(tokens)

			# Initialize arrays
			self.featureNeurons = self.initialiseFeatureNeuronsSequence(self.cs, self.fs)
			if(trainSparseConnectionsTensor and not inferenceMode):
				self.featureConnections = self.initialiseFeatureConnectionsSequenceSparse(self.cs, self.fs)
			else:
				self.featureConnections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

			# Populate arrays with data from observedColumnsDict (required for inference)
			if(inferenceMode):
				if(trainSequenceObservedColumnsMatchSequenceWords):
					self.populateArrays(tokens, self.sequenceObservedColumnsDict)
				else:
					self.populateArrays(tokens, self.observedColumnsDict2)
		else:
			self.featureWords = []
			self.featureIndicesInObservedTensor = pt.empty((0,), dtype=pt.long)
			self.fIdxTensor = pt.empty((0,), dtype=pt.long)
			self.fs = 0
			self.featureWordToIndex = {}
			self.indexToFeatureWord = {}
			self.columnStartIndicesTensor = None
			self.columnEndIndicesTensor = None
			self.columnFeatureLocalIndices = None
			if(conceptColumnsDelimitByPOS):
				GIAANNproto_sequenceConcepts.processConceptWords(self, 0, tokens, tokens)
			self.featureNeurons = None
			self.featureConnections = None

	def identifyObservedColumnFeatureWords(self, tokens, observedColumn):
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
			featureWords = []
			featureIndicesInObserved = []
			#print("\nidentifyObservedColumnFeatureWords: num words = ", len(tokens))
			for wordIndex, token in enumerate(tokens):
				featureWord = token.word
				featureLemma = token.lemma
				if(useDedicatedConceptNames and wordIndex in self.observedColumnsSequenceWordIndexDict):	
					if(useDedicatedConceptNames2):
						#only provide 1 observedColumn to identifyObservedColumnFeatureWords (therefore this condition will only be triggered once when when featureLemma == observedColumn.conceptName of some arbitrary concept column. Once triggered a singular artificial variablePrimeConceptFeatureNeuronName will be added)
						featureWords.append(variablePrimeConceptFeatureNeuronName)
					featureIndicesInObserved.append(featureIndexPrimeConceptNeuron)
					#print("concept node found = ", featureLemma)
				elif(featureWord in observedColumn.featureWordToIndex):
					featureWords.append(featureWord)
					featureIndicesInObserved.append(observedColumn.featureWordToIndex[featureWord])
			if(not trainSequenceObservedColumnsMatchSequenceWords):
				featureIndicesInObserved = self.removeDuplicates(featureIndicesInObserved)
				featureWords = self.removeDuplicates(featureWords)
			featureIndicesInObservedTensor = pt.tensor(featureIndicesInObserved, dtype=pt.long)
		else:
			featureWords = observedColumn.featureWordToIndex.keys()
			featureIndicesInObservedTensor = pt.tensor(list(observedColumn.featureWordToIndex.values()), dtype=pt.long)
		
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly and trainSequenceObservedColumnsMatchSequenceWords):
			fIdxTensor = pt.arange(len(featureWords), dtype=pt.long)
		else:
			featureWordToIndex = {}
			for idx, featureWord in enumerate(featureWords):
				featureWordToIndex[featureWord] = idx
			fIdxTensor = pt.tensor([featureWordToIndex[fw] for fw in featureWords], dtype=pt.long)
		
		return featureWords, featureIndicesInObservedTensor, fIdxTensor
		
	def getObservedColumnFeatureIndices(self):
		return self.featureIndicesInObservedTensor, self.fIdxTensor

	def ensureTokenConceptColumnIndexList(self):
		if(not hasattr(self, "tokenConceptColumnIndexList") or self.tokenConceptColumnIndexList is None or len(self.tokenConceptColumnIndexList) != len(self.tokens)):
			result = GIAANNproto_sequenceConcepts.processConceptWords(self, 0, self.tokens, self.tokens)
			if(result is None):
				raise RuntimeError("ensureTokenConceptColumnIndexList error: failed to compute token concept column assignment")
			if(not hasattr(self, "tokenConceptColumnIndexList") or self.tokenConceptColumnIndexList is None):
				raise RuntimeError("ensureTokenConceptColumnIndexList error: tokenConceptColumnIndexList was not generated")
			if(len(self.tokenConceptColumnIndexList) != len(self.tokens)):
				raise RuntimeError("ensureTokenConceptColumnIndexList error: tokenConceptColumnIndexList length mismatch")
		return

	def getTrainRequiredSourceFeatureIndicesByObservedColumnGeneric(self, observedColumnsByConceptIndex):
		result = {}
		for conceptIndex in observedColumnsByConceptIndex.keys():
			result[conceptIndex] = set()
		for tokenIndex, conceptIndex in enumerate(self.tokenConceptColumnIndexList):
			if(conceptIndex is None):
				raise RuntimeError(f"getTrainRequiredSourceFeatureIndicesByObservedColumnGeneric error: unassigned token index {tokenIndex}")
			normalisedConceptIndex = int(conceptIndex)
			observedColumn = observedColumnsByConceptIndex.get(normalisedConceptIndex)
			if(observedColumn is None):
				raise RuntimeError(f"getTrainRequiredSourceFeatureIndicesByObservedColumnGeneric error: missing observed column for conceptIndex {normalisedConceptIndex}")
			if(tokenIndex in self.columnsIndexSequenceWordIndexDict):
				sourceFeatureIndex = featureIndexPrimeConceptNeuron
			else:
				featureWord = self.tokens[tokenIndex].word
				if(featureWord not in observedColumn.featureWordToIndex):
					raise RuntimeError(f"getTrainRequiredSourceFeatureIndicesByObservedColumnGeneric error: feature word '{featureWord}' not found in observed column '{observedColumn.conceptName}'")
				sourceFeatureIndex = int(observedColumn.featureWordToIndex[featureWord])
			result[normalisedConceptIndex].add(int(sourceFeatureIndex))
		for conceptIndex, requiredSourceFeatureIndices in result.items():
			if(len(requiredSourceFeatureIndices) == 0):
				raise RuntimeError(f"getTrainRequiredSourceFeatureIndicesByObservedColumnGeneric error: no required source features for conceptIndex {conceptIndex}")
			result[conceptIndex] = sorted(requiredSourceFeatureIndices)
		return result

	def getTrainRequiredSourceFeatureIndicesByObservedColumn(self):
		self.ensureTokenConceptColumnIndexList()
		observedColumnsByConceptIndex = {}
		result = {}
		for observedColumn in self.observedColumnsDict.values():
			conceptIndex = int(observedColumn.conceptIndex)
			if(conceptIndex in observedColumnsByConceptIndex):
				raise RuntimeError(f"getTrainRequiredSourceFeatureIndicesByObservedColumn error: duplicate observed column conceptIndex {conceptIndex}")
			observedColumnsByConceptIndex[conceptIndex] = observedColumn
			result[conceptIndex] = []
		if(optimisationGetTrainRequiredSourceFeatureIndicesByObservedColumnVectorize and trainSequenceObservedColumnsUseSequenceFeaturesOnly and trainSequenceObservedColumnsMatchSequenceWords):
			if(any(conceptIndex is None for conceptIndex in self.tokenConceptColumnIndexList)):
				raise RuntimeError("getTrainRequiredSourceFeatureIndicesByObservedColumn error: tokenConceptColumnIndexList contains unassigned entries")
			conceptIndexTensor = pt.tensor(self.tokenConceptColumnIndexList, dtype=pt.long)
			featureIndexTensor = self.featureIndicesInObservedTensor.to(conceptIndexTensor.device)
			if(featureIndexTensor.dim() != 1):
				raise RuntimeError("getTrainRequiredSourceFeatureIndicesByObservedColumn error: featureIndicesInObservedTensor rank mismatch")
			if(featureIndexTensor.shape[0] != conceptIndexTensor.shape[0]):
				raise RuntimeError("getTrainRequiredSourceFeatureIndicesByObservedColumn error: featureIndicesInObservedTensor length mismatch")
			combinedKeys = conceptIndexTensor * self.databaseNetworkObject.f + featureIndexTensor
			sortedCombinedKeys = pt.sort(combinedKeys).values
			uniqueCombinedKeys = pt.unique_consecutive(sortedCombinedKeys)
			uniqueConceptIndices = pt.div(uniqueCombinedKeys, self.databaseNetworkObject.f, rounding_mode='floor')
			uniqueFeatureIndices = pt.remainder(uniqueCombinedKeys, self.databaseNetworkObject.f)
			for conceptIndexValue, featureIndexValue in zip(uniqueConceptIndices.tolist(), uniqueFeatureIndices.tolist()):
				if(conceptIndexValue not in result):
					raise RuntimeError(f"getTrainRequiredSourceFeatureIndicesByObservedColumn error: missing observed column for conceptIndex {conceptIndexValue}")
				result[conceptIndexValue].append(int(featureIndexValue))
			for conceptIndex, requiredSourceFeatureIndices in result.items():
				if(len(requiredSourceFeatureIndices) == 0):
					raise RuntimeError(f"getTrainRequiredSourceFeatureIndicesByObservedColumn error: no required source features for conceptIndex {conceptIndex}")
		else:
			result = self.getTrainRequiredSourceFeatureIndicesByObservedColumnGeneric(observedColumnsByConceptIndex)
		self.requiredSourceFeatureIndicesByObservedColumn = result
		return result
	
	def removeDuplicates(self, lst):
		#python requires ordered sets
		lst = list(dict.fromkeys(lst))
		return lst
		
	#@staticmethod
	def initialiseFeatureNeuronsSequence(self, cs, fs):
		featureNeurons = pt.zeros(self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, dtype=arrayType)
		return featureNeurons

	#@staticmethod
	def initialiseFeatureConnectionsSequence(self, cs, fs):
		featureConnections = pt.zeros(self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs, dtype=arrayType)
		return featureConnections

	#@staticmethod
	def initialiseFeatureConnectionsSequenceSparse(self, cs, fs):
		targetSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)
		emptyIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=deviceSparse)
		emptyValues = pt.empty((0,), dtype=arrayType, device=deviceSparse)
		featureConnections = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=targetSize, dtype=arrayType, device=deviceSparse)
		return featureConnections

	def useTrainSparseConnectionsTensor(self):
		result = trainSparseConnectionsTensor and self.featureConnections is not None and self.featureConnections.layout == pt.sparse_coo
		return result

	def coalesceSequenceFeatureConnectionsSparse(self, targetDevice=None):
		if(not self.useTrainSparseConnectionsTensor()):
			raise RuntimeError("coalesceSequenceFeatureConnectionsSparse error: train sparse connections tensor is disabled")
		result = self.featureConnections.coalesce()
		if(targetDevice is not None and result.device != targetDevice):
			result = result.to(targetDevice)
		return result

	def buildSequenceConnectionPropertyUpdateSparse(self, propertyIndex, propertyTensor):
		if(not self.useTrainSparseConnectionsTensor()):
			raise RuntimeError("buildSequenceConnectionPropertyUpdateSparse error: train sparse connections tensor is disabled")
		targetSize = self.featureConnections.size()
		propertyTargetSize = targetSize[1:]
		if(propertyIndex < 0 or propertyIndex >= targetSize[0]):
			raise RuntimeError(f"buildSequenceConnectionPropertyUpdateSparse error: propertyIndex {propertyIndex} out of range")
		if(propertyTensor is None):
			result = self.initialiseSparseTensor(targetSize, deviceSparse)
		else:
			if(tuple(propertyTensor.size()) != tuple(propertyTargetSize)):
				raise RuntimeError(f"buildSequenceConnectionPropertyUpdateSparse error: propertyTensor size {tuple(propertyTensor.size())} does not match expected size {tuple(propertyTargetSize)}")
			if(propertyTensor.layout == pt.sparse_coo):
				propertySparse = propertyTensor.coalesce()
				if(propertySparse.device != deviceSparse):
					propertySparse = propertySparse.to(deviceSparse)
			else:
				if(propertyTensor.device != deviceSparse):
					propertyTensor = propertyTensor.to(deviceSparse)
				propertySparse = propertyTensor.to_sparse().coalesce()
			if(propertySparse.dim() != len(propertyTargetSize)):
				raise RuntimeError(f"buildSequenceConnectionPropertyUpdateSparse error: propertyTensor rank {propertySparse.dim()} does not match expected rank {len(propertyTargetSize)}")
			propertyIndices = propertySparse.indices()
			propertyValues = propertySparse.values()
			propertyRow = pt.full((1, propertyIndices.shape[1]), propertyIndex, dtype=pt.long, device=propertyIndices.device)
			updateIndices = pt.cat([propertyRow, propertyIndices], dim=0)
			result = pt.sparse_coo_tensor(updateIndices, propertyValues, size=targetSize, dtype=arrayType, device=deviceSparse).coalesce()
		return result

	def removeSequenceConnectionPropertySparse(self, propertyIndex):
		if(not self.useTrainSparseConnectionsTensor()):
			raise RuntimeError("removeSequenceConnectionPropertySparse error: train sparse connections tensor is disabled")
		connectionSparse = self.coalesceSequenceFeatureConnectionsSparse()
		connectionIndices = connectionSparse.indices()
		connectionValues = connectionSparse.values()
		if(connectionIndices.numel() > 0):
			keepMask = connectionIndices[0] != propertyIndex
			filteredIndices = connectionIndices[:, keepMask]
			filteredValues = connectionValues[keepMask]
			self.featureConnections = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=connectionSparse.size(), dtype=arrayType, device=deviceSparse).coalesce()
		else:
			self.featureConnections = connectionSparse
		return

	def addSequenceConnectionPropertyUpdate(self, propertyIndex, propertyTensor):
		if(not self.useTrainSparseConnectionsTensor()):
			raise RuntimeError("addSequenceConnectionPropertyUpdate error: train sparse connections tensor is disabled")
		connectionTargetSparse = self.coalesceSequenceFeatureConnectionsSparse()
		connectionUpdateSparse = self.buildSequenceConnectionPropertyUpdateSparse(propertyIndex, propertyTensor)
		self.featureConnections = self.addSparseUpdateNonNegative(connectionTargetSparse, connectionUpdateSparse)
		return

	def setSequenceConnectionPropertyUpdate(self, propertyIndex, propertyTensor):
		if(not self.useTrainSparseConnectionsTensor()):
			raise RuntimeError("setSequenceConnectionPropertyUpdate error: train sparse connections tensor is disabled")
		self.removeSequenceConnectionPropertySparse(propertyIndex)
		if(propertyTensor is not None):
			self.addSequenceConnectionPropertyUpdate(propertyIndex, propertyTensor)
		return

	def transformSequenceConnectionPropertyValues(self, propertyIndex, transformType, transformValue=None):
		if(not self.useTrainSparseConnectionsTensor()):
			raise RuntimeError("transformSequenceConnectionPropertyValues error: train sparse connections tensor is disabled")
		connectionSparse = self.coalesceSequenceFeatureConnectionsSparse()
		connectionIndices = connectionSparse.indices()
		connectionValues = connectionSparse.values()
		updatedValues = connectionValues
		propertyMask = connectionIndices[0] == propertyIndex
		if(propertyMask.any()):
			updatedValues = connectionValues.clone()
			if(transformType == "clampMax"):
				updatedValues[propertyMask] = updatedValues[propertyMask].clamp(max=transformValue)
			elif(transformType == "tanh"):
				updatedValues[propertyMask] = pt.tanh(updatedValues[propertyMask])
			else:
				raise RuntimeError(f"transformSequenceConnectionPropertyValues error: unsupported transformType {transformType}")
		self.featureConnections = pt.sparse_coo_tensor(connectionIndices, updatedValues, size=connectionSparse.size(), dtype=arrayType, device=deviceSparse).coalesce()
		return

	def extractSequenceConnectionPropertySparse(self, propertyIndex):
		if(not self.useTrainSparseConnectionsTensor()):
			raise RuntimeError("extractSequenceConnectionPropertySparse error: train sparse connections tensor is disabled")
		connectionSparse = self.coalesceSequenceFeatureConnectionsSparse()
		connectionIndices = connectionSparse.indices()
		connectionValues = connectionSparse.values()
		propertyTargetSize = connectionSparse.size()[1:]
		if(connectionIndices.numel() > 0):
			propertyMask = connectionIndices[0] == propertyIndex
			filteredIndices = connectionIndices[1:, propertyMask]
			filteredValues = connectionValues[propertyMask]
		else:
			filteredIndices = pt.empty((len(propertyTargetSize), 0), dtype=pt.long, device=connectionSparse.device)
			filteredValues = pt.empty((0,), dtype=arrayType, device=connectionSparse.device)
		result = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=propertyTargetSize, dtype=arrayType, device=connectionSparse.device).coalesce()
		return result

	def computeColumnLocalFeatureMaps(self, tokens):
		if(not trainSequenceObservedColumnsMatchSequenceWords):
			return
		try:
			result = GIAANNproto_sequenceConcepts.processConceptWords(self, 0, tokens, tokens)
		except Exception:
			result = None
		if(not result):
			return
		conceptIndices, startIndices, endIndices = result
		if(startIndices is None or endIndices is None):
			return
		self.columnStartIndicesTensor = startIndices
		self.columnEndIndicesTensor = endIndices
		startList = startIndices.tolist()
		endList = endIndices.tolist()
		self.columnFeatureLocalIndices = []
		featureIndicesList = self.featureIndicesInObservedTensor.tolist()
		for cIdx in range(len(startList)):
			localMap = {}
			startIdx = max(0, startList[cIdx])
			endIdx = min(len(featureIndicesList), endList[cIdx])
			for localIndex in range(startIdx, endIdx):
				globalIndex = int(featureIndicesList[localIndex])
				if(globalIndex not in localMap):
					localMap[globalIndex] = []
				localMap[globalIndex].append(localIndex)
			self.columnFeatureLocalIndices.append(localMap)

	def mapGlobalToLocalIndices(self, defaultTensor, globalTensor, columnIndex, branchTensor=None):
		result = defaultTensor
		columnLocalMap = None
		conceptIndexKey = None
		if(self.columnFeatureLocalIndices is not None):
			if(columnIndex < len(self.columnFeatureLocalIndices)):
				columnLocalMap = self.columnFeatureLocalIndices[columnIndex]
		if(columnLocalMap is not None and columnLocalMap):
			defaultCPU = defaultTensor.detach().cpu()
			globalCPU = globalTensor.detach().cpu()
			branchCPU = branchTensor.detach().cpu() if branchTensor is not None else None
			defaultList = defaultCPU.tolist()
			globalList = globalCPU.tolist()
			branchList = branchCPU.tolist() if branchCPU is not None else None
			newList = []
			branchMappedCount = 0
			branchMappedNonZero = 0
			if(randomlyAssignBranches and branchList is not None):
				observedColumn = self.sequenceObservedColumnsDict.get(columnIndex) if trainSequenceObservedColumnsMatchSequenceWords else self.observedColumnsDict2.get(columnIndex)
				conceptIndexKey = observedColumn.conceptIndex if observedColumn is not None else None
			for idx, (defaultValue, globalValue) in enumerate(zip(defaultList, globalList)):
				candidates = columnLocalMap.get(int(globalValue))
				if(candidates and len(candidates) > 0):
					if(branchList is not None):
						branchMappedCount = branchMappedCount + 1
						branchIndex = int(branchList[idx])
						candidateIndex = branchIndex if branchIndex < len(candidates) else len(candidates) - 1
						if(randomlyAssignBranches and conceptIndexKey is not None):
							branchOrder = GIAANNproto_sequenceConcepts.buildDeterministicBranchOrder(conceptIndexKey, int(globalValue))
							if(branchIndex in branchOrder):
								candidateIndex = branchOrder.index(branchIndex)
								if(candidateIndex >= len(candidates)):
									candidateIndex = len(candidates) - 1
						if(candidateIndex > 0):
							branchMappedNonZero = branchMappedNonZero + 1
						newList.append(int(candidates[candidateIndex]))
					else:
						newList.append(int(candidates[0]))
				else:
					newList.append(int(defaultValue))
			result = pt.tensor(newList, dtype=defaultTensor.dtype, device=defaultTensor.device)
		return result
	
	def populateArrays(self, tokens, sequenceObservedColumnsDict):
		databaseNetworkObject = self.databaseNetworkObject
		#print("\n\n\n\n\npopulate_arrays:")
		
		# Optimized code for collecting indices and data for feature neurons
		cIdxList = []
		fIdxList = []
		featureListIndices = []
		featureListValues = []

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()
			if(useGPUsparseStrict and not useGPUsparse):
				featureIndicesInObserved = featureIndicesInObserved.to(deviceSparse)
				fIdxTensor = fIdxTensor.to(deviceSparse)
			numFeatures = len(fIdxTensor)

			cIdxList.append(pt.full((numFeatures,), cIdx, dtype=pt.long))
			fIdxList.append(fIdxTensor)

			if lowMem:
				featureNeurons = observedColumn.featureNeurons.coalesce()
			else:
				# Slice the globalFeatureNeurons as before
				featureNeurons = GIAANNproto_sparseTensors.sliceSparseTensor(self.databaseNetworkObject.globalFeatureNeurons, 3, observedColumn.conceptIndex)

			if(not useGPUsparseStrict and useGPUdense and not useGPUsparse):
				featureNeurons = featureNeurons.to(deviceDense)

			indices = featureNeurons.indices()  # [4, n_entries] for a 4D sparse tensor: (property, branch, segment, feature_idx)
			values = featureNeurons.values()

			# Ensure that featureIndicesInObserved is sorted if not already
			featureIndicesInObservedSorted, fIdxSortIdx = pt.sort(featureIndicesInObserved)
			fIdxTensorSorted = fIdxTensor[fIdxSortIdx]

			# Instead of expanding and comparing, directly check membership
			mask = pt.isin(indices[3], featureIndicesInObservedSorted)

			# Filter indices and values by mask
			filteredIndices = indices[:, mask]
			filteredValues = values[mask]

			if filteredIndices.size(1) > 0:
				# We need to find the corresponding f_idx for each filtered feature_idx.
				# Use searchsorted on the sorted featureIndicesInObserved
				positions = pt.searchsorted(featureIndicesInObservedSorted, filteredIndices[3])
				filteredFIdxTensor = fIdxTensorSorted[positions]
				filteredFIdxTensor = self.mapGlobalToLocalIndices(filteredFIdxTensor, filteredIndices[3], cIdx, filteredIndices[1])
			else:
				# If no matches, just create empty tensors that match the expected shape.
				filteredFIdxTensor = pt.empty((0,), dtype=fIdxTensor.dtype, device=fIdxTensor.device)

			# Adjust indices as in original code:
			# Original: filtered_indices = cat([filtered_indices[0:2], full_like(..., c_idx), filtered_indices[2:3]])
			# filteredIndices has shape [3, *], we insert a dimension for cIdx after the first two rows:
			# The final dimension order is: property, type, cIdx, feature_idx
			# Before insertion: filteredIndices = [property, type, feature_idx]
			# After insertion:  filteredIndices = [property, type, cIdx, feature_idx]
			if filteredIndices.size(1) > 0:
				filteredIndices[3] = filteredFIdxTensor
			# Insert cIdx row
			cIdxCol = pt.full((1, filteredIndices.size(1)), cIdx, dtype=pt.long, device=filteredIndices.device)
			filteredIndices = pt.cat([filteredIndices[0:3], cIdxCol, filteredIndices[3:4]], dim=0)

			if not useGPUsparse:
				filteredIndices = filteredIndices.to(deviceSparse)
				filteredValues = filteredValues.to(deviceSparse)

			featureListIndices.append(filteredIndices)
			featureListValues.append(filteredValues)

		# Combine results
		if featureListIndices:
			combinedIndices = pt.cat(featureListIndices, dim=1)
			combinedValues = pt.cat(featureListValues, dim=0)
			# Convert to dense as per original code, though consider keeping sparse for memory savings
			self.featureNeurons = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=self.featureNeurons.size(), dtype=arrayType, device=deviceDense).to_dense()

		# Now handle connections
		connectionIndicesList = []
		connectionValuesList = []

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()
			if(useGPUsparseStrict and not useGPUsparse):
				featureIndicesInObserved = featureIndicesInObserved.to(deviceSparse)
				fIdxTensor = fIdxTensor.to(deviceSparse)

			# Get indices and values from the sparse tensor
			featureConnections = observedColumn.materialiseFeatureConnections(loadAllStored=True, targetDevice=deviceSparse).coalesce()
			if(not useGPUsparseStrict and not useGPUsparse):
				featureConnections = featureConnections.to(deviceDense)

			indices = featureConnections.indices()  # shape [6, n_entries]
			values = featureConnections.values()	# shape [n_entries]

			# Sort featureIndicesInObserved and fIdxTensor together if not already sorted
			featureIndicesInObservedSorted, fIdxSortIdx = pt.sort(featureIndicesInObserved)
			fIdxTensorSorted = fIdxTensor[fIdxSortIdx]

			# For each other column
			for otherCIdx, otherObservedColumn in sequenceObservedColumnsDict.items():
				otherFeatureIndicesInObserved, otherFIdxTensor = self.getObservedColumnFeatureIndices()
				if(useGPUsparseStrict and not useGPUsparse):
					otherFeatureIndicesInObserved = otherFeatureIndicesInObserved.to(deviceSparse)
					otherFIdxTensor = otherFIdxTensor.to(deviceSparse)
				otherConceptIndex = otherObservedColumn.conceptIndex

				# Sort otherFeatureIndicesInObserved and otherFIdxTensor if not sorted
				otherFeatureIndicesInObservedSorted, otherFIdxSortIdx = pt.sort(otherFeatureIndicesInObserved)
				otherFIdxTensorSorted = otherFIdxTensor[otherFIdxSortIdx]

				# Create boolean masks directly:
				maskConcept = (indices[4] == otherConceptIndex)
				maskF2 = pt.isin(indices[3], featureIndicesInObservedSorted)
				maskF4 = pt.isin(indices[5], otherFeatureIndicesInObservedSorted)

				combinedMask = maskConcept & maskF2 & maskF4

				# Filter indices and values
				filteredIndices = indices[:, combinedMask]
				filteredValues = values[combinedMask]

				# If we got no matches, filteredIndices and filteredValues will be empty.
				# We do NOT continue here; we proceed to create and append empty results as per the original requirement.

				if filteredIndices.numel() > 0:
					sourceGlobalIndices = filteredIndices[3].clone()
					targetGlobalIndices = filteredIndices[5].clone()
					# Map indices[3] back to fIdxTensor
					fIdxPositions = pt.searchsorted(featureIndicesInObservedSorted, filteredIndices[3])
					mappedFIdx = fIdxTensorSorted[fIdxPositions]

					# Map indices[5] back to otherFIdxTensor
					otherFIdxPositions = pt.searchsorted(otherFeatureIndicesInObservedSorted, filteredIndices[5])
					mappedOtherFIdx = otherFIdxTensorSorted[otherFIdxPositions]

					mappedFIdx = self.mapGlobalToLocalIndices(mappedFIdx, sourceGlobalIndices, cIdx)
					mappedOtherFIdx = self.mapGlobalToLocalIndices(mappedOtherFIdx, targetGlobalIndices, otherCIdx, filteredIndices[1])

					# Adjust indices:
					# After filtering, we have:
					#   filteredIndices = [property, branch, segment, feature_idx, concept_idx, other_feature_idx]
					# We want to replace concept_idx with otherCIdx and feature_idx with mappedFIdx, other_feature_idx with mappedOtherFIdx.
					filteredIndices[3] = mappedFIdx
					filteredIndices[4] = otherCIdx
					filteredIndices[5] = mappedOtherFIdx
					if(debugDrawNeuronActivations):
						sampleCount = min(5, filteredIndices.shape[1])
						for sampleIdx in range(sampleCount):
							globalSourceIdx = int(sourceGlobalIndices[sampleIdx].item())
							localSourceIdx = int(mappedFIdx[sampleIdx].item())
							sourceFeatureName = self.indexToFeatureWord.get(localSourceIdx, "NA") if hasattr(self, "indexToFeatureWord") else "NA"
							globalTargetIdx = int(targetGlobalIndices[sampleIdx].item())
							localTargetIdx = int(mappedOtherFIdx[sampleIdx].item())
							targetFeatureName = self.indexToFeatureWord.get(localTargetIdx, "NA") if hasattr(self, "indexToFeatureWord") else "NA"
							
				else:
					# Even if empty, we need to maintain correct shape to append
					pass

				# Insert cIdx after the segment dimension as per the original code.
				cIdxCol = pt.full((1, filteredIndices.size(1)), cIdx, dtype=pt.long, device=filteredIndices.device)
				filteredIndices = pt.cat([filteredIndices[0:3], cIdxCol, filteredIndices[3:]], dim=0)

				if not useGPUsparse:
					filteredIndices = filteredIndices.to(deviceSparse)
					filteredValues = filteredValues.to(deviceSparse)

				connectionIndicesList.append(filteredIndices)
				connectionValuesList.append(filteredValues)

		# Combine results
		if connectionIndicesList:
			combinedIndices = pt.cat(connectionIndicesList, dim=1)
			combinedValues = pt.cat(connectionValuesList, dim=0)

			self.featureConnections = pt.sparse_coo_tensor(
				combinedIndices, combinedValues,
				size=self.featureConnections.size(),
				dtype=arrayType, 
				device=deviceDense
			).to_dense()
			if(debugDrawNeuronActivations):
				strengthSum = self.featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex].sum().item()
	
	def updateObservedColumnsWrapper(self, inference=False):
		self.debugInferenceActive = inference
		self.inferenceConceptUpdateCounts = {} if inference else None
		if(trainSequenceObservedColumnsMatchSequenceWords):
			#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
			self.updateObservedColumns(self.sequenceObservedColumnsDict, inference, mode="default")
		else:
			self.updateObservedColumns(self.observedColumnsDict2, inference, mode="default")

	def updateObservedColumns(self, sequenceObservedColumnsDict, inference, mode):
		if(arrayIndexPropertiesEfficient and not inference):
			updateObservedColumnsEfficientStartTime = None
			if(debugPrintTrainSectionTimes):
				updateObservedColumnsEfficientStartTime = time.perf_counter()
			self.updateObservedColumnsEfficient(sequenceObservedColumnsDict, mode)
			if(debugPrintTrainSectionTimes):
				GIAANNproto_debug.debugTrainSectionTimesAdd(self.databaseNetworkObject, "updateObservedColumnsEfficient", time.perf_counter() - updateObservedColumnsEfficientStartTime)
		else:
			self.updateObservedColumnsVerbose(sequenceObservedColumnsDict, mode)
	
	def updateObservedColumnsVerbose(self, sequenceObservedColumnsDict, mode):
		databaseNetworkObject = self.databaseNetworkObject
		# Update observed columns with data from sequence arrays
		if(GIAANNproto_debug.debugPrintGPUramUsage):
			if(executionMode=="train"):
				GIAANNproto_debug.debugPrintRamUsage("updateObservedColumnsVerbose:start", "mode = " + str(mode))

		inferenceConceptUpdateCounts = self.inferenceConceptUpdateCounts if getattr(self, "debugInferenceActive", False) else None
		featureNeuronsDelta = self.featureNeurons
		featureConnectionsDelta = self.featureConnections
		if(useGPUsparseStrict and not useGPUsparse):
			featureNeuronsDelta = featureNeuronsDelta.to(deviceSparse)
			if(not self.useTrainSparseConnectionsTensor()):
				featureConnectionsDelta = featureConnectionsDelta.to(deviceSparse)

		featureNeuronsDeltaSparse = featureNeuronsDelta.to_sparse()
		if(self.useTrainSparseConnectionsTensor()):
			featureConnectionsDeltaSparse = self.coalesceSequenceFeatureConnectionsSparse()
		else:
			featureConnectionsDeltaSparse = featureConnectionsDelta.to_sparse()
		featureNeuronsCurrentSparse = None
		featureConnectionsCurrentSparse = None

		replacePropertiesEnabled = arrayIndexPropertiesActivation or arrayIndexPropertiesTime	
		assert not replacePropertiesEnabled, "replacePropertiesEnabled is not robust to duplicate features"
		addPropertiesEnabled = arrayIndexPropertiesStrength or arrayIndexPropertiesPermanence
		
		if(replacePropertiesEnabled):
			featureNeuronsCurrentSparse = self.featureNeurons.to_sparse()
			if(self.useTrainSparseConnectionsTensor()):
				featureConnectionsCurrentSparse = featureConnectionsDeltaSparse
			else:
				featureConnectionsCurrentSparse = self.featureConnections.to_sparse()
		elif(arrayIndexPropertiesPos):
			featureNeuronsCurrentSparse = self.featureNeurons.to_sparse()
			if(self.useTrainSparseConnectionsTensor()):
				featureConnectionsCurrentSparse = featureConnectionsDeltaSparse
			else:
				featureConnectionsCurrentSparse = self.featureConnections.to_sparse()
		if not useGPUsparse:
			featureNeuronsDeltaSparse = featureNeuronsDeltaSparse.to(deviceSparse)
			featureConnectionsDeltaSparse = featureConnectionsDeltaSparse.to(deviceSparse)
			if(featureNeuronsCurrentSparse is not None):
				featureNeuronsCurrentSparse = featureNeuronsCurrentSparse.to(deviceSparse)
			if(featureConnectionsCurrentSparse is not None):
				featureConnectionsCurrentSparse = featureConnectionsCurrentSparse.to(deviceSparse)

		if(addPropertiesEnabled):
			addPropertyIndicesList = []
			if(arrayIndexPropertiesStrength):
				addPropertyIndicesList.append(databaseNetworkObject.arrayIndexPropertiesStrengthIndex)
			if(arrayIndexPropertiesPermanence):
				addPropertyIndicesList.append(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex)
			addPropertyIndices = pt.tensor(addPropertyIndicesList, dtype=pt.long)
			addPropertyMaskLookup = self.buildMaskLookup(self.databaseNetworkObject.arrayNumberOfProperties, addPropertyIndices.to(featureNeuronsDeltaSparse.device), featureNeuronsDeltaSparse.device)
		posPropertyMaskLookupFeature = None
		posPropertyMaskLookupConnection = None
		posPropertyIndices = None
		if(arrayIndexPropertiesPos):
			posPropertyIndices = pt.tensor([databaseNetworkObject.arrayIndexPropertiesPosIndex], dtype=pt.long)
			posPropertyMaskLookupFeature = self.buildMaskLookup(self.databaseNetworkObject.arrayNumberOfProperties, posPropertyIndices.to(featureNeuronsDeltaSparse.device), featureNeuronsDeltaSparse.device)
		if(replacePropertiesEnabled):
			replacePropertyMaskLookup = None
			replacePropertyIndicesList = []
			if(arrayIndexPropertiesActivationCreate):
				replacePropertyIndicesList.append(databaseNetworkObject.arrayIndexPropertiesActivationIndex)
			if(arrayIndexPropertiesTime):
				replacePropertyIndicesList.append(databaseNetworkObject.arrayIndexPropertiesTimeIndex)
			replacePropertyIndices = pt.tensor(replacePropertyIndicesList, dtype=pt.long)
			replacePropertyMaskLookup = self.buildMaskLookup(self.databaseNetworkObject.arrayNumberOfProperties, replacePropertyIndices.to(featureNeuronsDeltaSparse.device), featureNeuronsDeltaSparse.device)

		featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()
		fIdxTensorDevice = fIdxTensor.to(featureNeuronsDeltaSparse.device)
		featureIndicesObservedDevice = featureIndicesInObserved.to(featureNeuronsDeltaSparse.device)
		sequenceFeatureMaskLookup = self.buildMaskLookup(self.fs, fIdxTensorDevice, featureNeuronsDeltaSparse.device)
		observedFeatureMaskLookup = self.buildMaskLookup(self.databaseNetworkObject.f, featureIndicesObservedDevice, featureNeuronsDeltaSparse.device)

		connectionDevice = featureConnectionsDeltaSparse.device
		if(featureConnectionsCurrentSparse is not None):
			connectionDevice = featureConnectionsCurrentSparse.device
		sequenceConceptIndices = self.conceptIndicesInSequenceObservedTensor.to(connectionDevice)
		sequenceConceptIndicesUnique = pt.unique(sequenceConceptIndices)
		sequenceConceptMaskLookup = self.buildMaskLookup(self.databaseNetworkObject.c, sequenceConceptIndicesUnique, connectionDevice)
		if(arrayIndexPropertiesPos):
			posPropertyMaskLookupConnection = self.buildMaskLookup(self.databaseNetworkObject.arrayNumberOfProperties, posPropertyIndices.to(connectionDevice), connectionDevice)

		if not lowMem:
			globalFeatureNeurons = self.databaseNetworkObject.globalFeatureNeurons.coalesce()

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			conceptIndex = observedColumn.conceptIndex
			updateCount = None
			if(inferenceConceptUpdateCounts is not None):
				updateCount = inferenceConceptUpdateCounts.get(conceptIndex, 0) + 1
				inferenceConceptUpdateCounts[conceptIndex] = updateCount

			if lowMem:
				featureTargetSparse = observedColumn.featureNeurons.coalesce()
				featureTargetSize = featureTargetSparse.size()
			else:
				featureTargetSparse = globalFeatureNeurons
				featureTargetSize = featureTargetSparse.size()
			featureUpdatesPos = None
			connectionUpdatesPos = None

			if(addPropertiesEnabled):
				featureUpdatesAdd = self.extractSequenceFeatureUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureNeuronsDeltaSparse, addPropertyMaskLookup, sequenceFeatureMaskLookup, featureTargetSize, insertConceptIndex=None if lowMem else conceptIndex)
			if(replacePropertiesEnabled):
				featureUpdatesReplace = self.extractSequenceFeatureUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureNeuronsCurrentSparse, replacePropertyMaskLookup, sequenceFeatureMaskLookup, featureTargetSize, insertConceptIndex=None if lowMem else conceptIndex)
			if(arrayIndexPropertiesPos):
				featureUpdatesPos = self.extractSequenceFeatureUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureNeuronsCurrentSparse, posPropertyMaskLookupFeature, sequenceFeatureMaskLookup, featureTargetSize, insertConceptIndex=None if lowMem else conceptIndex)

			if(replacePropertiesEnabled):
				activationUpdateBranches = None
				preserveActivationOnReplace = False
				if(self.debugInferenceActive and multipleDendriticBranches and updateCount is not None and updateCount > 1 and databaseNetworkObject.arrayIndexPropertiesActivationIndex is not None):
					preserveActivationOnReplace = True
				if(self.debugInferenceActive and multipleDendriticBranches and databaseNetworkObject.arrayIndexPropertiesActivationIndex is not None):
					activationUpdates = featureUpdatesReplace.coalesce()
					activationUpdateIndices = activationUpdates.indices()
					activationUpdateMask = (activationUpdateIndices[0] == databaseNetworkObject.arrayIndexPropertiesActivationIndex)
					if(not lowMem):
						activationUpdateMask = activationUpdateMask & (activationUpdateIndices[3] == conceptIndex)
					if(activationUpdateMask.any()):
						activationUpdateBranches = pt.unique(activationUpdateIndices[1][activationUpdateMask])
				featureTargetSparse = featureTargetSparse.coalesce()
				targetIndices = featureTargetSparse.indices()
				targetValues = featureTargetSparse.values()
				if lowMem:
					removeMask = replacePropertyMaskLookup[targetIndices[0]] & observedFeatureMaskLookup[targetIndices[3]]
				else:
					removeMask = replacePropertyMaskLookup[targetIndices[0]] & (targetIndices[3] == conceptIndex) & observedFeatureMaskLookup[targetIndices[4]]
				if(preserveActivationOnReplace):
					removeMask = removeMask & (targetIndices[0] != databaseNetworkObject.arrayIndexPropertiesActivationIndex)
				elif(self.debugInferenceActive and multipleDendriticBranches and databaseNetworkObject.arrayIndexPropertiesActivationIndex is not None):
					if(activationUpdateBranches is not None and activationUpdateBranches.numel() > 0):
						activationBranchLookup = self.buildMaskLookup(numberOfDendriticBranches, activationUpdateBranches.to(targetIndices.device), targetIndices.device)
						activationMask = (targetIndices[0] == databaseNetworkObject.arrayIndexPropertiesActivationIndex)
						if(lowMem):
							activationMask = activationMask & observedFeatureMaskLookup[targetIndices[3]]
						else:
							activationMask = activationMask & (targetIndices[3] == conceptIndex) & observedFeatureMaskLookup[targetIndices[4]]
						activationBranchMask = activationMask & activationBranchLookup[targetIndices[1]]
						removeMask = (removeMask & pt.logical_not(activationMask)) | activationBranchMask
					else:
						removeMask = removeMask & (targetIndices[0] != databaseNetworkObject.arrayIndexPropertiesActivationIndex)
				keepMask = pt.logical_not(removeMask)
				filteredTargetIndices = targetIndices[:, keepMask]
				filteredTargetValues = targetValues[keepMask]
				featureTargetSparse = pt.sparse_coo_tensor(filteredTargetIndices, filteredTargetValues, size=featureTargetSize, dtype=arrayType, device=deviceSparse)

				combinedFeatureUpdates = self.combineSparseUpdates(featureUpdatesAdd, featureUpdatesReplace, featureTargetSize)
			else:
				combinedFeatureUpdates = featureUpdatesAdd
			featureTargetSparse = self.addSparseUpdate(featureTargetSparse, combinedFeatureUpdates)
			if(arrayIndexPropertiesPos):
				featureTargetSparse = self.applySparseMaxUpdate(featureTargetSparse, featureUpdatesPos)

			if lowMem:
				observedColumn.featureNeurons = featureTargetSparse
			else:
				globalFeatureNeurons = featureTargetSparse

			connectionTargetSize = observedColumn.getFeatureConnectionsTargetSize()
			connectionMaterialisedTargetSize = observedColumn.getMaterialisedFeatureConnectionsTargetSize()
			connectionUpdatesAddBySourceFeature = {}
			connectionUpdatesReplaceBySourceFeature = {}
			connectionUpdatesPosBySourceFeature = {}
			if(addPropertiesEnabled):
				connectionUpdatesAdd = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsDeltaSparse, addPropertyMaskLookup, sequenceFeatureMaskLookup, connectionMaterialisedTargetSize)
				connectionUpdatesAddBySourceFeature = self.splitConnectionUpdateSparseBySourceFeature(connectionUpdatesAdd, connectionTargetSize)
			if(replacePropertiesEnabled):
				connectionUpdatesReplace = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsCurrentSparse, replacePropertyMaskLookup, sequenceFeatureMaskLookup, connectionMaterialisedTargetSize)
				connectionUpdatesReplaceBySourceFeature = self.splitConnectionUpdateSparseBySourceFeature(connectionUpdatesReplace, connectionTargetSize)
			if(arrayIndexPropertiesPos):
				connectionUpdatesPos = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsCurrentSparse, posPropertyMaskLookupConnection, sequenceFeatureMaskLookup, connectionMaterialisedTargetSize)
				connectionUpdatesPosBySourceFeature = self.splitConnectionUpdateSparseBySourceFeature(connectionUpdatesPos, connectionTargetSize)
			connectionSourceFeatureIndices = self.getVerboseConnectionSourceFeatureIndices(observedColumn, conceptIndex, [connectionUpdatesAddBySourceFeature, connectionUpdatesReplaceBySourceFeature, connectionUpdatesPosBySourceFeature])
			for sourceFeatureIndex in connectionSourceFeatureIndices:
				connectionTargetSparse = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=deviceSparse, createMissing=False).coalesce()
				connectionSourceTargetSize = connectionTargetSparse.size()
				emptyConnectionUpdates = self.initialiseEmptySparseTensor(connectionSourceTargetSize, connectionTargetSparse.device)
				connectionUpdatesAddSource = emptyConnectionUpdates
				connectionUpdatesReplaceSource = emptyConnectionUpdates
				connectionUpdatesPosSource = emptyConnectionUpdates
				if(addPropertiesEnabled):
					connectionUpdatesAddSource = connectionUpdatesAddBySourceFeature.get(sourceFeatureIndex, emptyConnectionUpdates)
				if(replacePropertiesEnabled):
					connectionUpdatesReplaceSource = connectionUpdatesReplaceBySourceFeature.get(sourceFeatureIndex, emptyConnectionUpdates)
				if(arrayIndexPropertiesPos):
					connectionUpdatesPosSource = connectionUpdatesPosBySourceFeature.get(sourceFeatureIndex, emptyConnectionUpdates)
				if(replacePropertiesEnabled):
					connectionIndices = connectionTargetSparse.indices()
					connectionValues = connectionTargetSparse.values()
					removeMaskConnections = replacePropertyMaskLookup[connectionIndices[0]]
					removeMaskConnections = removeMaskConnections & sequenceConceptMaskLookup[connectionIndices[3]]
					removeMaskConnections = removeMaskConnections & observedFeatureMaskLookup[connectionIndices[4]]
					keepConnectionsMask = pt.logical_not(removeMaskConnections)
					filteredConnectionIndices = connectionIndices[:, keepConnectionsMask]
					filteredConnectionValues = connectionValues[keepConnectionsMask]
					connectionTargetSparse = pt.sparse_coo_tensor(filteredConnectionIndices, filteredConnectionValues, size=connectionSourceTargetSize, dtype=arrayType, device=deviceSparse)
					combinedConnectionUpdates = self.combineSparseUpdates(connectionUpdatesAddSource, connectionUpdatesReplaceSource, connectionSourceTargetSize)
				else:
					combinedConnectionUpdates = connectionUpdatesAddSource
				if(addPropertiesEnabled):
					connectionTargetSparse = self.addSparseUpdate(connectionTargetSparse, combinedConnectionUpdates)
				if(arrayIndexPropertiesPos):
					connectionTargetSparse = self.applySparseMaxUpdate(connectionTargetSparse, connectionUpdatesPosSource)
				observedColumn.setFeatureConnectionsForSourceFeature(sourceFeatureIndex, connectionTargetSparse)

		if not lowMem:
			self.databaseNetworkObject.globalFeatureNeurons = globalFeatureNeurons
		if(GIAANNproto_debug.debugPrintGPUramUsage):
			if(executionMode=="train"):
				GIAANNproto_debug.debugPrintRamUsage("updateObservedColumnsVerbose:end", "mode = " + str(mode))

	def updateObservedColumnsEfficient(self, sequenceObservedColumnsDict, mode):
		databaseNetworkObject = self.databaseNetworkObject
		if(GIAANNproto_debug.debugPrintGPUramUsage):
			if(executionMode=="train"):
				GIAANNproto_debug.debugPrintRamUsage("updateObservedColumnsEfficient:start", "mode = " + str(mode))
		if not arrayIndexPropertiesStrength:
			return

		featureNeuronsDelta = self.featureNeurons[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
		featureConnectionsDelta = None
		featureConnectionsDeltaSparse = None
		if(self.useTrainSparseConnectionsTensor()):
			featureConnectionsDeltaSparse = self.extractSequenceConnectionPropertySparse(databaseNetworkObject.arrayIndexPropertiesStrengthIndex)
		else:
			featureConnectionsDelta = self.featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
		if(useGPUsparseStrict and not useGPUsparse):
			featureNeuronsDelta = featureNeuronsDelta.to(deviceSparse)
			if(featureConnectionsDelta is not None):
				featureConnectionsDelta = featureConnectionsDelta.to(deviceSparse)

		featureNeuronsDeltaSparse = featureNeuronsDelta.to_sparse()
		if(featureConnectionsDelta is not None):
			featureConnectionsDeltaSparse = featureConnectionsDelta.to_sparse()
		if not useGPUsparse:
			featureNeuronsDeltaSparse = featureNeuronsDeltaSparse.to(deviceSparse)
			featureConnectionsDeltaSparse = featureConnectionsDeltaSparse.to(deviceSparse)

		featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()
		featureDevice = featureNeuronsDeltaSparse.device
		connectionDevice = featureConnectionsDeltaSparse.device
		featureIndicesObservedFeatureDevice = featureIndicesInObserved.to(featureDevice)
		featureIndicesObservedConnectionDevice = featureIndicesInObserved.to(connectionDevice)
		sequenceFeatureMaskFeature = None
		sequenceFeatureMaskConnection = None
		if(fIdxTensor.numel() != self.fs):
			fIdxTensorFeatureDevice = fIdxTensor.to(featureDevice)
			sequenceFeatureMaskFeature = self.buildMaskLookup(self.fs, fIdxTensorFeatureDevice, featureDevice)
			if(connectionDevice == featureDevice):
				sequenceFeatureMaskConnection = sequenceFeatureMaskFeature
			else:
				fIdxTensorConnectionDevice = fIdxTensor.to(connectionDevice)
				sequenceFeatureMaskConnection = self.buildMaskLookup(self.fs, fIdxTensorConnectionDevice, connectionDevice)

		featureIndices = featureNeuronsDeltaSparse.indices()
		featureValues = featureNeuronsDeltaSparse.values()
		featureIndices, featureValues = self.filterSparseByFeatureMask(featureIndices, featureValues, sequenceFeatureMaskFeature, [3])

		connectionIndices = featureConnectionsDeltaSparse.indices()
		connectionValues = featureConnectionsDeltaSparse.values()
		connectionIndices, connectionValues = self.filterSparseByFeatureMask(connectionIndices, connectionValues, sequenceFeatureMaskConnection, [3, 5])

		connectionSourceCombinedKeys = None
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			getObservedColumnsByConceptIndexStartTime = None
			if(debugPrintTrainSectionTimes):
				getObservedColumnsByConceptIndexStartTime = time.perf_counter()
		observedColumnsByConceptIndex = self.getObservedColumnsByConceptIndex(sequenceObservedColumnsDict)
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			if(debugPrintTrainSectionTimes):
				GIAANNproto_debug.debugTrainSectionTimesAdd(self.databaseNetworkObject, "updateObservedColumnsEfficient:getObservedColumnsByConceptIndex", time.perf_counter() - getObservedColumnsByConceptIndexStartTime)
		conceptIndicesFeatureTensor = self.conceptIndicesInSequenceObservedTensor.to(featureDevice)
		conceptIndicesConnectionTensor = self.conceptIndicesInSequenceObservedTensor.to(connectionDevice)

		#A: update feature neurons;
		if(optimisationArrayIndexPropertiesEfficientSerialNeurons):
			featureRanges, featureIndicesSorted, featureValuesSorted = self.buildSparseColumnRanges(featureIndices, featureValues, 2)
			
			globalFeatureNeurons = None
			globalFeatureNeuronUpdates = None
			if(not lowMem):
				globalFeatureNeurons = self.databaseNetworkObject.globalFeatureNeurons
				if(optimisationCombineSparseUpdatesPerSequence):
					globalFeatureNeuronUpdates = []
			
			for cIdx, observedColumn in sequenceObservedColumnsDict.items():
				conceptIndex = observedColumn.conceptIndex

				#A: update feature neurons;
				if(lowMem):
					featureTargetSparse = observedColumn.featureNeurons
					featureTargetSize = featureTargetSparse.size()
				else:
					featureTargetSparse = globalFeatureNeurons
					featureTargetSize = globalFeatureNeurons.size()
				featureRange = featureRanges.get(cIdx)
				if(featureRange is not None):
					start, end = featureRange
					featureUpdateIndices = featureIndicesSorted[:, start:end]
					featureUpdateValues = featureValuesSorted[start:end]
					featureUpdates = self.buildFeaturePropertyUpdateSparse(featureUpdateIndices, featureUpdateValues, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, featureIndicesObservedFeatureDevice, featureTargetSize, insertConceptIndex=None if lowMem else conceptIndex)
					if(lowMem):
						featureTargetSparse = self.addSparseUpdateNonNegative(featureTargetSparse, featureUpdates)
					else:
						if(optimisationCombineSparseUpdatesPerSequence):
							globalFeatureNeuronUpdates.append(featureUpdates)
						else:
							featureTargetSparse = self.addSparseUpdateNonNegative(featureTargetSparse, featureUpdates)
				if(lowMem):
					observedColumn.featureNeurons = featureTargetSparse
				else:
					if(not optimisationCombineSparseUpdatesPerSequence):
						globalFeatureNeurons = featureTargetSparse

			if(not lowMem):
				if(optimisationCombineSparseUpdatesPerSequence):
					if(globalFeatureNeuronUpdates is None):
						raise RuntimeError("updateObservedColumnsEfficient error: globalFeatureNeuronUpdates is None while optimisationCombineSparseUpdatesPerSequence")
					if(len(globalFeatureNeuronUpdates) > 0):
						combinedFeatureUpdates = self.combineSparseUpdatesList(globalFeatureNeuronUpdates, globalFeatureNeurons.size())
						globalFeatureNeurons = self.addSparseUpdateNonNegative(globalFeatureNeurons, combinedFeatureUpdates)
				self.databaseNetworkObject.globalFeatureNeurons = globalFeatureNeurons
		else:
			if(featureIndices.numel() > 0):
				if(lowMem):
					featureConceptIndicesUnique = pt.unique(conceptIndicesFeatureTensor[featureIndices[2]], sorted=True)
					featureTargetSize = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, featureConceptIndicesUnique.shape[0], databaseNetworkObject.f)
					featureTargetSparse = self.gatherFeatureNeuronConceptBucketTensor(observedColumnsByConceptIndex, featureConceptIndicesUnique, featureDevice)
					featureUpdates = self.buildFeaturePropertyUpdateSparseBatched(featureIndices, featureValues, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, featureIndicesObservedFeatureDevice, conceptIndicesFeatureTensor, featureTargetSize, featureConceptIndicesUnique)
					featureTargetSparse = self.addSparseUpdateNonNegative(featureTargetSparse, featureUpdates)
					self.scatterFeatureNeuronConceptBucketTensor(observedColumnsByConceptIndex, featureConceptIndicesUnique, featureTargetSparse)
				else:
					globalFeatureNeurons = self.databaseNetworkObject.globalFeatureNeurons
					featureUpdates = self.buildFeaturePropertyUpdateSparseBatched(featureIndices, featureValues, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, featureIndicesObservedFeatureDevice, conceptIndicesFeatureTensor, globalFeatureNeurons.size())
					globalFeatureNeurons = self.addSparseUpdateNonNegative(globalFeatureNeurons, featureUpdates)
					self.databaseNetworkObject.globalFeatureNeurons = globalFeatureNeurons

		#B: update feature connections;
		if(optimisationArrayIndexPropertiesEfficientSerialConnections):
			connectionStorageDevice = self.getConnectionSerialStorageDevice()

			connectionRanges, connectionIndicesSorted, connectionValuesSorted = self.buildSparseColumnRanges(connectionIndices, connectionValues, 2)

			for cIdx, observedColumn in sequenceObservedColumnsDict.items():
				connectionRange = connectionRanges.get(cIdx)
				if(connectionRange is not None):
					connectionTargetSize = observedColumn.getFeatureConnectionsTargetSize()
					connectionTargetsBySourceFeature = {}
					start, end = connectionRange
					connectionUpdateIndices = connectionIndicesSorted[:, start:end]
					connectionUpdateValues = connectionValuesSorted[start:end]
					self.applyConnectionSourceFeaturePropertyUpdates(connectionTargetsBySourceFeature, observedColumn, connectionUpdateIndices, connectionUpdateValues, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, featureIndicesObservedConnectionDevice, conceptIndicesConnectionTensor, connectionTargetSize, connectionDevice, connectionStorageDevice)
					updatedSourceFeatureIndices = sorted(connectionTargetsBySourceFeature.keys())
					for sourceFeatureIndex in updatedSourceFeatureIndices:
						observedColumn.setFeatureConnectionsForSourceFeature(sourceFeatureIndex, connectionTargetsBySourceFeature[sourceFeatureIndex])
					if(len(updatedSourceFeatureIndices) > 0):
						combinedUpdatedSourceFeatureIndices = sorted(set(observedColumn.getTrainPreparedSourceFeatureIndices() + updatedSourceFeatureIndices))
						observedColumn.setTrainPreparedSourceFeatureIndices(combinedUpdatedSourceFeatureIndices)
		else:
			if(connectionIndices.numel() > 0):
				connectionSourceCombinedKeys = self.buildConnectionSourceCombinedKeys(connectionIndices, featureIndicesObservedConnectionDevice, conceptIndicesConnectionTensor)
			if(connectionSourceCombinedKeys is not None and connectionSourceCombinedKeys.numel() > 0):
				connectionSourceCombinedKeysUnique = pt.unique(connectionSourceCombinedKeys, sorted=True)
				connectionTargetSize = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, connectionSourceCombinedKeysUnique.shape[0], databaseNetworkObject.c, databaseNetworkObject.f)
				connectionTargetSparse = self.gatherConnectionSourceBucketTensor(observedColumnsByConceptIndex, connectionSourceCombinedKeysUnique, connectionDevice)
				if(connectionIndices.numel() > 0):
					connectionUpdates = self.buildConnectionSourceBucketUpdateSparse(connectionIndices, connectionValues, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, featureIndicesObservedConnectionDevice, conceptIndicesConnectionTensor, connectionSourceCombinedKeysUnique, connectionTargetSize)
					connectionTargetSparse = self.addSparseUpdateNonNegative(connectionTargetSparse, connectionUpdates)
				self.scatterConnectionSourceBucketTensor(observedColumnsByConceptIndex, connectionSourceCombinedKeysUnique, connectionTargetSparse)
			
		if(GIAANNproto_debug.debugPrintGPUramUsage):
			if(executionMode=="train"):
				GIAANNproto_debug.debugPrintRamUsage("updateObservedColumnsEfficient:end", "mode = " + str(mode))

	def getObservedColumnsByConceptIndex(self, sequenceObservedColumnsDict):
		result = {}
		for observedColumn in sequenceObservedColumnsDict.values():
			conceptIndex = int(observedColumn.conceptIndex)
			if(conceptIndex in result):
				if(result[conceptIndex] is not observedColumn):
					raise RuntimeError(f"getObservedColumnsByConceptIndex error: conflicting observed column for conceptIndex {conceptIndex}")
			else:
				result[conceptIndex] = observedColumn
		return result

	def initialiseSparseTensor(self, targetSize, targetDevice):
		emptyIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=targetDevice)
		emptyValues = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=targetSize, dtype=arrayType, device=targetDevice)
		return result

	def buildFeaturePropertyUpdateSparseBatched(self, indices, values, propertyIndex, featureIndicesInObserved, conceptIndicesTensor, targetSize, compactConceptIndices=None):
		result = self.initialiseSparseTensor(targetSize, indices.device)
		if(indices.numel() > 0):
			branch = indices[0]
			segment = indices[1]
			conceptIndex = conceptIndicesTensor[indices[2]]
			featureIndex = indices[3]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				featureIndex = featureIndicesInObserved[featureIndex]
			if(compactConceptIndices is not None):
				conceptIndex = pt.searchsorted(compactConceptIndices, conceptIndex)
			propertyRow = pt.full_like(segment, propertyIndex)
			updateIndices = pt.stack((propertyRow, branch, segment, conceptIndex, featureIndex), dim=0)
			result = pt.sparse_coo_tensor(updateIndices, values, size=targetSize, dtype=arrayType, device=indices.device)
		return result

	def gatherFeatureNeuronConceptBucketTensor(self, observedColumnsByConceptIndex, compactConceptIndices, targetDevice):
		targetSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, compactConceptIndices.shape[0], self.databaseNetworkObject.f)
		combinedIndicesList = []
		combinedValuesList = []
		conceptIndexList = compactConceptIndices.detach().cpu().tolist()
		for conceptBucketIndex, conceptIndexValue in enumerate(conceptIndexList):
			if(int(conceptIndexValue) not in observedColumnsByConceptIndex):
				raise RuntimeError(f"gatherFeatureNeuronConceptBucketTensor error: missing observed column for conceptIndex {int(conceptIndexValue)}")
			featureTargetSparse = observedColumnsByConceptIndex[int(conceptIndexValue)].featureNeurons
			if(featureTargetSparse.device != targetDevice):
				featureTargetSparse = featureTargetSparse.to(targetDevice)
			featureTargetSparse = featureTargetSparse.coalesce()
			if(featureTargetSparse._nnz() > 0):
				featureTargetIndices = featureTargetSparse.indices()
				featureTargetValues = featureTargetSparse.values()
				conceptBucketRow = pt.full((1, featureTargetIndices.shape[1]), conceptBucketIndex, dtype=pt.long, device=featureTargetIndices.device)
				batchedIndices = pt.cat([featureTargetIndices[0:3], conceptBucketRow, featureTargetIndices[3:]], dim=0)
				combinedIndicesList.append(batchedIndices)
				combinedValuesList.append(featureTargetValues)
		if(len(combinedIndicesList) > 0):
			combinedIndices = pt.cat(combinedIndicesList, dim=1)
			combinedValues = pt.cat(combinedValuesList, dim=0)
			result = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSize, dtype=arrayType, device=targetDevice)
		else:
			result = self.initialiseSparseTensor(targetSize, targetDevice)
		return result

	def scatterFeatureNeuronConceptBucketTensor(self, observedColumnsByConceptIndex, compactConceptIndices, featureTargetSparse):
		featureTargetSparse = featureTargetSparse.coalesce()
		featureTargetIndices = featureTargetSparse.indices()
		featureTargetValues = featureTargetSparse.values()
		featureTargetSortedIndices = featureTargetIndices
		featureTargetSortedValues = featureTargetValues
		featureTargetBucketRanges = {}
		if(featureTargetIndices.numel() > 0):
			sortedBucketIndices, sortOrder = pt.sort(featureTargetIndices[3])
			featureTargetSortedIndices = featureTargetIndices[:, sortOrder]
			featureTargetSortedValues = featureTargetValues.index_select(0, sortOrder)
			uniqueBuckets, counts = pt.unique_consecutive(sortedBucketIndices, return_counts=True)
			starts = pt.cumsum(counts, 0) - counts
			for conceptBucketIndexValue, start, count in zip(uniqueBuckets.tolist(), starts.tolist(), counts.tolist()):
				featureTargetBucketRanges[int(conceptBucketIndexValue)] = (int(start), int(start + count))
		conceptIndexList = compactConceptIndices.detach().cpu().tolist()
		sourceTensorSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, self.databaseNetworkObject.f)
		for conceptBucketIndex, conceptIndexValue in enumerate(conceptIndexList):
			if(int(conceptIndexValue) not in observedColumnsByConceptIndex):
				raise RuntimeError(f"scatterFeatureNeuronConceptBucketTensor error: missing observed column for conceptIndex {int(conceptIndexValue)}")
			if(conceptBucketIndex in featureTargetBucketRanges):
				start, end = featureTargetBucketRanges[conceptBucketIndex]
				sourceIndices = pt.stack((featureTargetSortedIndices[0, start:end], featureTargetSortedIndices[1, start:end], featureTargetSortedIndices[2, start:end], featureTargetSortedIndices[4, start:end]), dim=0)
				sourceValues = featureTargetSortedValues[start:end]
				sourceTensor = pt.sparse_coo_tensor(sourceIndices, sourceValues, size=sourceTensorSize, dtype=arrayType, device=featureTargetSparse.device)
			else:
				sourceTensor = self.initialiseSparseTensor(sourceTensorSize, featureTargetSparse.device)
			observedColumnsByConceptIndex[int(conceptIndexValue)].featureNeurons = sourceTensor.coalesce()
		return

	def buildConnectionSourceCombinedKeys(self, indices, featureIndicesInObserved, conceptIndicesTensor):
		result = pt.empty((0,), dtype=pt.long, device=indices.device)
		if(indices.numel() > 0):
			sourceConceptIndex = conceptIndicesTensor[indices[2]]
			sourceFeatureIndex = indices[3]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				sourceFeatureIndex = featureIndicesInObserved[sourceFeatureIndex]
			result = sourceConceptIndex * self.databaseNetworkObject.f + sourceFeatureIndex
		return result

	def buildConnectionSourceBucketUpdateSparse(self, indices, values, propertyIndex, featureIndicesInObserved, conceptIndicesTensor, sourceCombinedKeysUnique, targetSize):
		result = self.initialiseSparseTensor(targetSize, indices.device)
		if(indices.numel() > 0):
			branch = indices[0]
			segment = indices[1]
			sourceConceptIndex = conceptIndicesTensor[indices[2]]
			sourceFeatureIndex = indices[3]
			targetConceptIndex = conceptIndicesTensor[indices[4]]
			targetFeatureIndex = indices[5]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				sourceFeatureIndex = featureIndicesInObserved[sourceFeatureIndex]
				targetFeatureIndex = featureIndicesInObserved[targetFeatureIndex]
			sourceCombinedKeys = sourceConceptIndex * self.databaseNetworkObject.f + sourceFeatureIndex
			sourceBucketIndex = pt.searchsorted(sourceCombinedKeysUnique, sourceCombinedKeys)
			propertyRow = pt.full_like(segment, propertyIndex)
			updateIndices = pt.stack((propertyRow, branch, segment, sourceBucketIndex, targetConceptIndex, targetFeatureIndex), dim=0)
			result = pt.sparse_coo_tensor(updateIndices, values, size=targetSize, dtype=arrayType, device=indices.device)
		return result

	def gatherConnectionSourceBucketTensor(self, observedColumnsByConceptIndex, sourceCombinedKeysUnique, targetDevice):
		targetSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, sourceCombinedKeysUnique.shape[0], self.databaseNetworkObject.c, self.databaseNetworkObject.f)
		result = None
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			gatherConnectionSourceBucketTensorStartTime = None
			if(debugPrintTrainSectionTimes):
				gatherConnectionSourceBucketTensorStartTime = time.perf_counter()
				GIAANNproto_debug.debugTrainSectionTimesContextPush(self.databaseNetworkObject, "updateObservedColumnsEfficient:gatherConnectionSourceBucketTensor")
		combinedIndicesList = []
		combinedValuesList = []
		sourceConceptIndexList = pt.div(sourceCombinedKeysUnique, self.databaseNetworkObject.f, rounding_mode='floor').detach().cpu().tolist()
		sourceFeatureIndexList = pt.remainder(sourceCombinedKeysUnique, self.databaseNetworkObject.f).detach().cpu().tolist()
		for sourceBucketIndex, (conceptIndexValue, sourceFeatureIndexValue) in enumerate(zip(sourceConceptIndexList, sourceFeatureIndexList)):
			if(int(conceptIndexValue) not in observedColumnsByConceptIndex):
				raise RuntimeError(f"gatherConnectionSourceBucketTensor error: missing observed column for conceptIndex {int(conceptIndexValue)}")
			sourceTensor = observedColumnsByConceptIndex[int(conceptIndexValue)].getFeatureConnectionsForSourceFeature(int(sourceFeatureIndexValue), targetDevice=targetDevice, createMissing=False)
			sourceTensor = sourceTensor.coalesce()
			if(sourceTensor._nnz() > 0):
				sourceIndices = sourceTensor.indices()
				sourceValues = sourceTensor.values()
				sourceBucketRow = pt.full((1, sourceIndices.shape[1]), sourceBucketIndex, dtype=pt.long, device=sourceIndices.device)
				batchedIndices = pt.cat([sourceIndices[0:3], sourceBucketRow, sourceIndices[3:]], dim=0)
				combinedIndicesList.append(batchedIndices)
				combinedValuesList.append(sourceValues)
		if(len(combinedIndicesList) > 0):
			combinedIndices = pt.cat(combinedIndicesList, dim=1)
			combinedValues = pt.cat(combinedValuesList, dim=0)
			result = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSize, dtype=arrayType, device=targetDevice)
		else:
			result = self.initialiseSparseTensor(targetSize, targetDevice)
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			if(debugPrintTrainSectionTimes):
				GIAANNproto_debug.debugTrainSectionTimesContextPop(self.databaseNetworkObject)
				GIAANNproto_debug.debugTrainSectionTimesAdd(self.databaseNetworkObject, "updateObservedColumnsEfficient:gatherConnectionSourceBucketTensor", time.perf_counter() - gatherConnectionSourceBucketTensorStartTime)
		return result

	def scatterConnectionSourceBucketTensor(self, observedColumnsByConceptIndex, sourceCombinedKeysUnique, connectionTargetSparse):
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			scatterConnectionSourceBucketTensorStartTime = None
			if(debugPrintTrainSectionTimes):
				scatterConnectionSourceBucketTensorStartTime = time.perf_counter()
				GIAANNproto_debug.debugTrainSectionTimesContextPush(self.databaseNetworkObject, "updateObservedColumnsEfficient:scatterConnectionSourceBucketTensor")
		connectionTargetSparse = connectionTargetSparse.coalesce()
		connectionTargetIndices = connectionTargetSparse.indices()
		connectionTargetValues = connectionTargetSparse.values()
		connectionTargetSortedIndices = connectionTargetIndices
		connectionTargetSortedValues = connectionTargetValues
		connectionTargetBucketRanges = {}
		if(connectionTargetIndices.numel() > 0):
			sortedBucketIndices, sortOrder = pt.sort(connectionTargetIndices[3])
			connectionTargetSortedIndices = connectionTargetIndices[:, sortOrder]
			connectionTargetSortedValues = connectionTargetValues.index_select(0, sortOrder)
			uniqueBuckets, counts = pt.unique_consecutive(sortedBucketIndices, return_counts=True)
			starts = pt.cumsum(counts, 0) - counts
			for sourceBucketIndexValue, start, count in zip(uniqueBuckets.tolist(), starts.tolist(), counts.tolist()):
				connectionTargetBucketRanges[int(sourceBucketIndexValue)] = (int(start), int(start + count))
		sourceConceptIndexList = pt.div(sourceCombinedKeysUnique, self.databaseNetworkObject.f, rounding_mode='floor').detach().cpu().tolist()
		sourceFeatureIndexList = pt.remainder(sourceCombinedKeysUnique, self.databaseNetworkObject.f).detach().cpu().tolist()
		sourceTensorSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, self.databaseNetworkObject.c, self.databaseNetworkObject.f)
		for sourceBucketIndex, (conceptIndexValue, sourceFeatureIndexValue) in enumerate(zip(sourceConceptIndexList, sourceFeatureIndexList)):
			if(int(conceptIndexValue) not in observedColumnsByConceptIndex):
				raise RuntimeError(f"scatterConnectionSourceBucketTensor error: missing observed column for conceptIndex {int(conceptIndexValue)}")
			observedColumn = observedColumnsByConceptIndex[int(conceptIndexValue)]
			if(not storeDatabaseInRam and observedColumn.hasTrainPreparedSourceFeatureIndices()):
				if(int(sourceFeatureIndexValue) not in observedColumn.trainPreparedSourceFeatureIndices):
					raise RuntimeError(f"scatterConnectionSourceBucketTensor error: sourceFeatureIndex {int(sourceFeatureIndexValue)} was not prepared for conceptIndex {int(conceptIndexValue)}")
			if(sourceBucketIndex in connectionTargetBucketRanges):
				start, end = connectionTargetBucketRanges[sourceBucketIndex]
				sourceIndices = pt.stack((connectionTargetSortedIndices[0, start:end], connectionTargetSortedIndices[1, start:end], connectionTargetSortedIndices[2, start:end], connectionTargetSortedIndices[4, start:end], connectionTargetSortedIndices[5, start:end]), dim=0)
				sourceValues = connectionTargetSortedValues[start:end]
				sourceTensor = pt.sparse_coo_tensor(sourceIndices, sourceValues, size=sourceTensorSize, dtype=arrayType, device=connectionTargetSparse.device)
			else:
				sourceTensor = self.initialiseSparseTensor(sourceTensorSize, connectionTargetSparse.device)
			observedColumn.setFeatureConnectionsForSourceFeature(int(sourceFeatureIndexValue), sourceTensor)
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			if(debugPrintTrainSectionTimes):
				GIAANNproto_debug.debugTrainSectionTimesContextPop(self.databaseNetworkObject)
				GIAANNproto_debug.debugTrainSectionTimesAdd(self.databaseNetworkObject, "updateObservedColumnsEfficient:scatterConnectionSourceBucketTensor", time.perf_counter() - scatterConnectionSourceBucketTensorStartTime)
		return

	def buildMaskLookup(self, maskSize, indices, device):
		maskLookup = pt.zeros((maskSize,), dtype=pt.bool, device=device)
		if(indices.numel() > 0):
			maskLookup[indices] = True
		return maskLookup

	def combineSparseUpdates(self, updateA, updateB, targetSize):
		updateA = updateA.coalesce()
		updateB = updateB.coalesce()
		indicesA = updateA.indices()
		valuesA = updateA.values()
		indicesB = updateB.indices()
		valuesB = updateB.values()
		if(indicesA.numel() == 0 and indicesB.numel() == 0):
			combinedIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=deviceSparse)
			combinedValues = pt.empty((0,), dtype=arrayType, device=deviceSparse)
		elif(indicesA.numel() == 0):
			combinedIndices = indicesB
			combinedValues = valuesB
		elif(indicesB.numel() == 0):
			combinedIndices = indicesA
			combinedValues = valuesA
		else:
			combinedIndices = pt.cat([indicesA, indicesB], dim=1)
			combinedValues = pt.cat([valuesA, valuesB], dim=0)
		combinedSparse = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSize, dtype=arrayType, device=deviceSparse)
		return combinedSparse
	
	if(optimisationCombineSparseUpdatesPerSequence):
		def combineSparseUpdatesList(self, updatesList, targetSize):
			combinedIndices = None
			combinedValues = None
			if(len(updatesList) > 0):
				indicesList = []
				valuesList = []
				for updateSparse in updatesList:
					updateSparse = updateSparse.coalesce()
					indicesList.append(updateSparse.indices())
					valuesList.append(updateSparse.values())
				if(len(indicesList) > 0):
					combinedIndices = pt.cat(indicesList, dim=1)
				else:
					combinedIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=deviceSparse)
				if(len(valuesList) > 0):
					combinedValues = pt.cat(valuesList, dim=0)
				else:
					combinedValues = pt.empty((0,), dtype=arrayType, device=deviceSparse)
			else:
				combinedIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=deviceSparse)
				combinedValues = pt.empty((0,), dtype=arrayType, device=deviceSparse)
			combinedSparse = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSize, dtype=arrayType, device=deviceSparse)
			return combinedSparse

	def addSparseUpdate(self, targetSparse, updateSparse):
		targetSparse = targetSparse.coalesce()
		updateSparse = updateSparse.coalesce()
		indicesTarget = targetSparse.indices()
		valuesTarget = targetSparse.values()
		indicesUpdate = updateSparse.indices()
		valuesUpdate = updateSparse.values()
		if(indicesUpdate.numel() > 0):
			combinedIndices = pt.cat([indicesTarget, indicesUpdate], dim=1)
			combinedValues = pt.cat([valuesTarget, valuesUpdate], dim=0)
			targetSparse = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSparse.size(), dtype=arrayType, device=deviceSparse)
			targetSparse = targetSparse.coalesce()
			targetSparse.values().clamp_(min=0)
		return targetSparse

	def validateCUDAObservedColumnUpdateInputs(self, targetSparse, updateSparse):
		if(optimisationUseCUDAObservedColumnUpdateKernel):
			if(targetSparse.device.type != "cuda"):
				raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: targetSparse must be CUDA tensor")
			if(updateSparse.device.type != "cuda"):
				raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: updateSparse must be CUDA tensor")
			if(targetSparse.dtype != pt.float32):
				raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: targetSparse dtype must be float32")
			if(updateSparse.dtype != pt.float32):
				raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: updateSparse dtype must be float32")
			if(targetSparse.layout != pt.sparse_coo):
				raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: targetSparse must be sparse COO tensor")
			if(updateSparse.layout != pt.sparse_coo):
				raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: updateSparse must be sparse COO tensor")
			if(targetSparse.dim() != updateSparse.dim()):
				raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: targetSparse and updateSparse must have matching rank")
		else:
			raise RuntimeError("validateCUDAObservedColumnUpdateInputs error: optimisationUseCUDAObservedColumnUpdateKernel is disabled")
		return

	def recordCUDAObservedColumnUpdateStats(self, stats):
		if(optimisationUseCUDAObservedColumnUpdateKernel):
			if(not hasattr(self.databaseNetworkObject, "cudaObservedColumnUpdateInstrumentation")):
				self.databaseNetworkObject.cudaObservedColumnUpdateInstrumentation = {"hashHitCount": 0, "overflowCount": 0, "rebuildCount": 0, "updateCalls": 0, "averageUpdateLatencySeconds": 0.0, "hashHitRate": 0.0}
			instrumentation = self.databaseNetworkObject.cudaObservedColumnUpdateInstrumentation
			instrumentation["hashHitCount"] = instrumentation["hashHitCount"] + int(stats["hash_hit_count"])
			instrumentation["overflowCount"] = instrumentation["overflowCount"] + int(stats["overflow_count"])
			instrumentation["rebuildCount"] = instrumentation["rebuildCount"] + int(stats["rebuild_count"])
			instrumentation["updateCalls"] = instrumentation["updateCalls"] + int(stats["update_calls"])
			instrumentation["averageUpdateLatencySeconds"] = float(stats["average_update_latency_seconds"])
			instrumentation["hashHitRate"] = float(stats["hash_hit_rate"])
		return

	def addSparseUpdateNonNegativeCUDA(self, targetSparse, updateSparse):
		resultSparse = targetSparse
		if(optimisationUseCUDAObservedColumnUpdateKernel):
			if not targetSparse.is_coalesced():
				targetSparse = targetSparse.coalesce()
			if not updateSparse.is_coalesced():
				updateSparse = updateSparse.coalesce()
			self.validateCUDAObservedColumnUpdateInputs(targetSparse, updateSparse)
			targetNNZ = max(1, int(targetSparse._nnz()))
			updateNNZ = int(updateSparse._nnz())
			overflowCapacityMultiplier = max(1.0, float(updateNNZ) / float(targetNNZ))
			accumulator = GIAANNproto_cudaObservedColumnUpdate.build_sparse_accumulator(targetSparse.indices(), targetSparse.values(), targetSparse.size(), overflowCapacityMultiplier=overflowCapacityMultiplier)
			accumulator = GIAANNproto_cudaObservedColumnUpdate.accumulate_sparse_updates(accumulator, updateSparse.indices(), updateSparse.values())
			exportIndices, exportValues, exportStats = GIAANNproto_cudaObservedColumnUpdate.export_coo(accumulator)
			self.recordCUDAObservedColumnUpdateStats(exportStats)
			resultSparse = pt.sparse_coo_tensor(exportIndices, exportValues, size=targetSparse.size(), dtype=arrayType, device=targetSparse.device)
		else:
			raise RuntimeError("addSparseUpdateNonNegativeCUDA error: optimisationUseCUDAObservedColumnUpdateKernel is disabled")
		return resultSparse

	def addSparseUpdateNonNegative(self, targetSparse, updateSparse):
		resultSparse = targetSparse
		if(updateSparse._nnz() > 0):
			if(optimisationUseCUDAObservedColumnUpdateKernel):
				resultSparse = self.addSparseUpdateNonNegativeCUDA(targetSparse, updateSparse)
			else:
				if not targetSparse.is_coalesced():
					targetSparse = targetSparse.coalesce()
				if not updateSparse.is_coalesced():
					updateSparse = updateSparse.coalesce()
				indicesTarget = targetSparse.indices()
				valuesTarget = targetSparse.values()
				indicesUpdate = updateSparse.indices()
				valuesUpdate = updateSparse.values()
				combinedIndices = pt.cat([indicesTarget, indicesUpdate], dim=1)
				combinedValues = pt.cat([valuesTarget, valuesUpdate], dim=0)
				combinedSparse = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSparse.size(), dtype=arrayType, device=deviceSparse)
				resultSparse = combinedSparse.coalesce()
		return resultSparse

	def filterSparseByFeatureMask(self, indices, values, featureMaskLookup, featureDims):
		if(featureMaskLookup is None or indices.numel() == 0):
			return indices, values
		mask = featureMaskLookup[indices[featureDims[0]]]
		for dim in featureDims[1:]:
			mask = mask & featureMaskLookup[indices[dim]]
		if(mask.any()):
			return indices[:, mask], values[mask]
		emptyIndices = pt.empty((indices.size(0), 0), dtype=pt.long, device=indices.device)
		emptyValues = pt.empty((0,), dtype=values.dtype, device=values.device)
		return emptyIndices, emptyValues

	def buildSparseColumnRanges(self, indices, values, columnDim):
		if(indices.numel() == 0):
			return {}, indices, values
		columnIndices = indices[columnDim]
		sortedColumnIndices, sortOrder = pt.sort(columnIndices)
		sortedIndices = indices[:, sortOrder]
		sortedValues = values[sortOrder]
		uniqueColumns, counts = pt.unique_consecutive(sortedColumnIndices, return_counts=True)
		starts = pt.cumsum(counts, 0) - counts
		ranges = {}
		for columnIndex, start, count in zip(uniqueColumns.tolist(), starts.tolist(), counts.tolist()):
			if(count > 0):
				ranges[columnIndex] = (start, start + count)
		return ranges, sortedIndices, sortedValues

	def buildFeaturePropertyUpdateSparse(self, indices, values, propertyIndex, featureIndicesInObserved, targetSize, insertConceptIndex=None):
		if(indices.numel() == 0):
			emptyIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=indices.device)
			emptyValues = pt.empty((0,), dtype=arrayType, device=indices.device)
			return pt.sparse_coo_tensor(emptyIndices, emptyValues, size=targetSize, dtype=arrayType, device=deviceSparse)
		branch = indices[0]
		segment = indices[1]
		featureIndex = indices[3]
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
			featureIndex = featureIndicesInObserved[featureIndex]
		propertyRow = pt.full_like(segment, propertyIndex)
		if(insertConceptIndex is None):
			updateIndices = pt.stack((propertyRow, branch, segment, featureIndex), dim=0)
		else:
			conceptRow = pt.full_like(segment, insertConceptIndex)
			updateIndices = pt.stack((propertyRow, branch, segment, conceptRow, featureIndex), dim=0)
		return pt.sparse_coo_tensor(updateIndices, values, size=targetSize, dtype=arrayType, device=deviceSparse)

	def buildConnectionPropertyUpdateSparse(self, indices, values, propertyIndex, featureIndicesInObserved, conceptIndicesTensor, targetSize):
		if(indices.numel() == 0):
			emptyIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=indices.device)
			emptyValues = pt.empty((0,), dtype=arrayType, device=indices.device)
			return pt.sparse_coo_tensor(emptyIndices, emptyValues, size=targetSize, dtype=arrayType, device=deviceSparse)
		branch = indices[0]
		segment = indices[1]
		sourceFeatureIndex = indices[3]
		targetConceptIndex = indices[4]
		targetFeatureIndex = indices[5]
		targetConceptIndex = conceptIndicesTensor[targetConceptIndex]
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
			sourceFeatureIndex = featureIndicesInObserved[sourceFeatureIndex]
			targetFeatureIndex = featureIndicesInObserved[targetFeatureIndex]
		propertyRow = pt.full_like(segment, propertyIndex)
		updateIndices = pt.stack((propertyRow, branch, segment, sourceFeatureIndex, targetConceptIndex, targetFeatureIndex), dim=0)
		return pt.sparse_coo_tensor(updateIndices, values, size=targetSize, dtype=arrayType, device=deviceSparse)

	def getConnectionSerialStorageDevice(self):
		result = pt.device("cpu")
		return result

	def applyConnectionSourceFeaturePropertyUpdates(self, connectionTargetsBySourceFeature, observedColumn, indices, values, propertyIndex, featureIndicesInObserved, conceptIndicesTensor, connectionTargetSize, connectionDevice, connectionStorageDevice=None):
		if(indices.numel() > 0):
			branch = indices[0]
			segment = indices[1]
			sourceFeatureIndex = indices[3]
			targetConceptIndex = indices[4]
			targetFeatureIndex = indices[5]
			targetConceptIndex = conceptIndicesTensor[targetConceptIndex]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				sourceFeatureIndex = featureIndicesInObserved[sourceFeatureIndex]
				targetFeatureIndex = featureIndicesInObserved[targetFeatureIndex]
			sortedSourceFeatureIndex, sortOrder = pt.sort(sourceFeatureIndex)
			sortedBranch = branch.index_select(0, sortOrder)
			sortedSegment = segment.index_select(0, sortOrder)
			sortedTargetConceptIndex = targetConceptIndex.index_select(0, sortOrder)
			sortedTargetFeatureIndex = targetFeatureIndex.index_select(0, sortOrder)
			sortedValues = values.index_select(0, sortOrder)
			uniqueSourceFeatures, counts = pt.unique_consecutive(sortedSourceFeatureIndex, return_counts=True)
			starts = pt.cumsum(counts, 0) - counts
			for sourceFeatureIndexValue, start, count in zip(uniqueSourceFeatures.tolist(), starts.tolist(), counts.tolist()):
				end = start + count
				groupBranch = sortedBranch[start:end]
				groupSegment = sortedSegment[start:end]
				groupTargetConceptIndex = sortedTargetConceptIndex[start:end]
				groupTargetFeatureIndex = sortedTargetFeatureIndex[start:end]
				groupValues = sortedValues[start:end]
				propertyRow = pt.full_like(groupSegment, propertyIndex)
				updateIndices = pt.stack((propertyRow, groupBranch, groupSegment, groupTargetConceptIndex, groupTargetFeatureIndex), dim=0)
				updateSparse = pt.sparse_coo_tensor(updateIndices, groupValues, size=connectionTargetSize, dtype=arrayType, device=indices.device)
				normalisedSourceFeatureIndex = int(sourceFeatureIndexValue)
				if(normalisedSourceFeatureIndex in connectionTargetsBySourceFeature):
					connectionTargetSparse = connectionTargetsBySourceFeature[normalisedSourceFeatureIndex]
					if(connectionTargetSparse.device != connectionDevice):
						connectionTargetSparse = connectionTargetSparse.to(connectionDevice)
				else:
					connectionTargetSparse = observedColumn.prepareFeatureConnectionsForSourceFeature(normalisedSourceFeatureIndex, targetDevice=connectionDevice, createMissing=False)
				connectionTargetSparse = self.addSparseUpdateNonNegative(connectionTargetSparse, updateSparse)
				if(connectionStorageDevice is not None):
					if(connectionTargetSparse.device != connectionStorageDevice):
						connectionTargetSparse = connectionTargetSparse.to(connectionStorageDevice)
				connectionTargetsBySourceFeature[normalisedSourceFeatureIndex] = connectionTargetSparse
		return

	def initialiseEmptySparseTensor(self, targetSize, targetDevice):
		emptyIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=targetDevice)
		emptyValues = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=targetSize, dtype=arrayType, device=targetDevice)
		return result

	def splitConnectionUpdateSparseBySourceFeature(self, updateSparse, targetSize):
		result = {}
		if(updateSparse is not None):
			updateSparse = updateSparse.coalesce()
			updateIndices = updateSparse.indices()
			updateValues = updateSparse.values()
			if(updateIndices.numel() > 0):
				sortedSourceFeatureIndices, sortOrder = pt.sort(updateIndices[3])
				sortedIndices = updateIndices[:, sortOrder]
				sortedValues = updateValues.index_select(0, sortOrder)
				uniqueSourceFeatures, counts = pt.unique_consecutive(sortedSourceFeatureIndices, return_counts=True)
				starts = pt.cumsum(counts, 0) - counts
				for sourceFeatureIndexValue, start, count in zip(uniqueSourceFeatures.tolist(), starts.tolist(), counts.tolist()):
					end = start + count
					sourceIndices = pt.stack((sortedIndices[0, start:end], sortedIndices[1, start:end], sortedIndices[2, start:end], sortedIndices[4, start:end], sortedIndices[5, start:end]), dim=0)
					sourceValues = sortedValues[start:end]
					result[int(sourceFeatureIndexValue)] = pt.sparse_coo_tensor(sourceIndices, sourceValues, size=targetSize, dtype=arrayType, device=updateSparse.device)
		return result

	def getVerboseConnectionSourceFeatureIndices(self, observedColumn, conceptIndex, connectionUpdatesBySourceFeatureList):
		resultSet = set()
		if(not getattr(self, "debugInferenceActive", False) and self.requiredSourceFeatureIndicesByObservedColumn is not None):
			if(conceptIndex not in self.requiredSourceFeatureIndicesByObservedColumn):
				raise RuntimeError(f"getVerboseConnectionSourceFeatureIndices error: missing required source features for conceptIndex {conceptIndex}")
			requiredSourceFeatureIndices = set(int(sourceFeatureIndex) for sourceFeatureIndex in self.requiredSourceFeatureIndicesByObservedColumn[conceptIndex])
		else:
			requiredSourceFeatureIndices = None
			if(storeDatabaseInRam):
				for sourceFeatureIndex in observedColumn.featureConnectionsBySourceFeature.keys():
					resultSet.add(int(sourceFeatureIndex))
			else:
				for sourceFeatureIndex in observedColumn.featureConnectionsBySourceFeature.keys():
					resultSet.add(int(sourceFeatureIndex))
		for connectionUpdatesBySourceFeature in connectionUpdatesBySourceFeatureList:
			for sourceFeatureIndex in connectionUpdatesBySourceFeature.keys():
				normalisedSourceFeatureIndex = int(sourceFeatureIndex)
				if(requiredSourceFeatureIndices is not None):
					if(normalisedSourceFeatureIndex not in requiredSourceFeatureIndices):
						raise RuntimeError(f"getVerboseConnectionSourceFeatureIndices error: unexpected source feature {normalisedSourceFeatureIndex} for conceptIndex {conceptIndex}")
				resultSet.add(normalisedSourceFeatureIndex)
		result = sorted(resultSet)
		return result

	def flattenSparseIndices(self, indices, size):
		device = indices.device
		sizeTensor = pt.tensor(size, dtype=pt.long, device=device)
		strides = pt.ones((len(size),), dtype=pt.long, device=device)
		for i in range(len(size)-2, -1, -1):
			strides[i] = strides[i+1] * sizeTensor[i+1]
		linear = (indices * strides.unsqueeze(1)).sum(dim=0)
		return linear

	def applySparseMaxUpdate(self, targetSparse, updateSparse):
		updatedSparse = targetSparse.coalesce()
		updateSparse = updateSparse.coalesce()
		targetIndices = updatedSparse.indices()
		targetValues = updatedSparse.values()
		updateIndices = updateSparse.indices()
		updateValues = updateSparse.values()
		if(updateIndices.numel() > 0):
			targetLinear = self.flattenSparseIndices(targetIndices, updatedSparse.size())
			updateLinear = self.flattenSparseIndices(updateIndices, updatedSparse.size())
			sortedTargetLinear, sortOrder = pt.sort(targetLinear)
			if(sortedTargetLinear.numel() > 0):
				updatePos = pt.searchsorted(sortedTargetLinear, updateLinear)
				inBounds = updatePos < sortedTargetLinear.numel()
				clampedPos = updatePos.clamp(max=sortedTargetLinear.numel()-1)
				matchMask = inBounds & (sortedTargetLinear[clampedPos] == updateLinear)
			else:
				updatePos = pt.zeros_like(updateLinear, dtype=pt.long)
				matchMask = pt.zeros_like(updateLinear, dtype=pt.bool)

			updatedTargetValues = targetValues
			if(matchMask.any()):
				matchedPositions = updatePos[matchMask]
				matchedTargetIndices = sortOrder[matchedPositions]
				updatedTargetValues = targetValues.clone()
				updatedTargetValues[matchedTargetIndices] = pt.maximum(updatedTargetValues[matchedTargetIndices], updateValues[matchMask])

			nonMatchMask = pt.logical_not(matchMask)
			if(nonMatchMask.any()):
				newIndices = updateIndices[:, nonMatchMask]
				newValues = updateValues[nonMatchMask]
				combinedIndices = pt.cat([targetIndices, newIndices], dim=1)
				combinedValues = pt.cat([updatedTargetValues, newValues], dim=0)
			else:
				combinedIndices = targetIndices
				combinedValues = updatedTargetValues

			updatedSparse = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=updatedSparse.size(), dtype=arrayType, device=deviceSparse)
		return updatedSparse

	def extractSequenceFeatureUpdates(self, cIdx, fIdxTensor, featureIndicesInObserved, featureNeuronsSparse, propertyMaskLookup, sequenceFeatureMaskLookup, targetSize, insertConceptIndex=None):
		indices = featureNeuronsSparse.indices()
		values = featureNeuronsSparse.values()
		mask = (indices[3] == cIdx)
		if(propertyMaskLookup is not None):
			mask = mask & propertyMaskLookup[indices[0]]
		if(sequenceFeatureMaskLookup is not None):
			mask = mask & sequenceFeatureMaskLookup[indices[4]]
		filteredIndices = indices[:, mask]
		filteredValues = values[mask]
		if(filteredIndices.numel() > 0):
			filteredIndices = pt.stack((
				filteredIndices[0],
				filteredIndices[1],
				filteredIndices[2],
				filteredIndices[4]
			), dim=0)
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				filteredIndices[3] = featureIndicesInObserved[filteredIndices[3]]
			if(insertConceptIndex is not None):
				conceptIndexRow = pt.full((1, filteredIndices.size(1)), insertConceptIndex, dtype=pt.long, device=filteredIndices.device)
				filteredIndices = pt.cat([filteredIndices[0:3], conceptIndexRow, filteredIndices[3:4]], dim=0)
		else:
			filteredIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=featureNeuronsSparse.device)
			filteredValues = pt.empty((0,), dtype=arrayType, device=featureNeuronsSparse.device)
		if not useGPUsparse:
			filteredIndices = filteredIndices.to(deviceSparse)
			filteredValues = filteredValues.to(deviceSparse)
		updateSparse = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=targetSize, dtype=arrayType, device=deviceSparse)
		return updateSparse

	def extractSequenceConnectionUpdates(self, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsSparse, propertyMaskLookup, sequenceFeatureMaskLookup, targetSize):
		indices = featureConnectionsSparse.indices()
		values = featureConnectionsSparse.values()
		mask = (indices[3] == cIdx)
		if(propertyMaskLookup is not None):
			mask = mask & propertyMaskLookup[indices[0]]
		if(sequenceFeatureMaskLookup is not None):
			mask = mask & sequenceFeatureMaskLookup[indices[4]]
			mask = mask & sequenceFeatureMaskLookup[indices[6]]
		filteredIndices = indices[:, mask]
		filteredValues = values[mask]
		if(filteredIndices.numel() > 0):
			filteredIndices = pt.stack((
				filteredIndices[0],
				filteredIndices[1],
				filteredIndices[2],
				filteredIndices[4],
				filteredIndices[5],
				filteredIndices[6]
			), dim=0)
			conceptIndicesTensor = self.conceptIndicesInSequenceObservedTensor.to(filteredIndices.device)
			filteredIndices[4] = conceptIndicesTensor[filteredIndices[4]]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				filteredIndices[3] = featureIndicesInObserved[filteredIndices[3]]
				filteredIndices[5] = featureIndicesInObserved[filteredIndices[5]]
		else:
			filteredIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=featureConnectionsSparse.device)
			filteredValues = pt.empty((0,), dtype=arrayType, device=featureConnectionsSparse.device)
		if not useGPUsparse:
			filteredIndices = filteredIndices.to(deviceSparse)
			filteredValues = filteredValues.to(deviceSparse)
		updateSparse = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=targetSize, dtype=arrayType, device=deviceSparse)
		return updateSparse
