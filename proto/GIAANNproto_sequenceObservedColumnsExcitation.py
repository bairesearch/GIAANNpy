"""GIAANNproto_sequenceObservedColumnsExcitation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto sequence Observed Columns Excitation

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_sequenceConcepts

def debugPrintConnectionSamples(label, indices, values, maxSamples=5):
	numEntries = indices.shape[1]
	print(f"\tsequenceObservedColumns debug: {label}: entries={numEntries}")
	if(numEntries == 0):
		return
	sampleCount = min(numEntries, maxSamples)
	for entryIndex in range(sampleCount):
		indexTuple = indices[:, entryIndex].tolist()
		value = float(values[entryIndex].item())
		print(f"\tsequenceObservedColumns debug:\tindices={indexTuple}, value={value}")

# Define the SequenceObservedColumns class
class SequenceObservedColumns:
	"""
	Contains sequence observed columns object arrays which stack a feature subset of the observed columns object arrays for the current sequence.
	"""
	def __init__(self, databaseNetworkObject, tokens, observedColumnsDict, observedColumnsSequenceWordIndexDict):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.databaseNetworkObject = databaseNetworkObject
		self.observedColumnsDict = observedColumnsDict	# key: lemma, value: ObservedColumn
		self.observedColumnsSequenceWordIndexDict = observedColumnsSequenceWordIndexDict	# key: sequence word index, value: ObservedColumn

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
		self.featureConnections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

		self.featureNeuronsOriginal = self.featureNeurons.clone()
		self.featureConnectionsOriginal = self.featureConnections.clone()

		# Populate arrays with data from observedColumnsDict (required for inference)
		if(useInference):
			if(trainSequenceObservedColumnsMatchSequenceWords):
				self.populateArrays(tokens, self.sequenceObservedColumnsDict)
			else:
				self.populateArrays(tokens, self.observedColumnsDict2)

			
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
						#only provide 1 observedColumn to identifyObservedColumnFeatureWords (therefore this condition will only be triggered once when when featureLemma == observedColumn.conceptName of some arbitrary concept column. Once triggered a singular artificial variableConceptNeuronFeatureName will be added)
						featureWords.append(variableConceptNeuronFeatureName)
					featureIndicesInObserved.append(featureIndexConceptNeuron)
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
	
	def removeDuplicates(self, lst):
		#python requires ordered sets
		lst = list(dict.fromkeys(lst))
		return lst
				
	@staticmethod
	def initialiseFeatureNeuronsSequence(cs, fs):
		featureNeurons = pt.zeros(arrayNumberOfProperties, arrayNumberOfSegments, cs, fs, dtype=arrayType)
		return featureNeurons

	@staticmethod
	def initialiseFeatureConnectionsSequence(cs, fs):
		featureConnections = pt.zeros(arrayNumberOfProperties, arrayNumberOfSegments, cs, fs, cs, fs, dtype=arrayType)
		return featureConnections

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

	def mapGlobalToLocalIndices(self, defaultTensor, globalTensor, columnIndex):
		if(self.columnFeatureLocalIndices is None):
			return defaultTensor
		if(columnIndex >= len(self.columnFeatureLocalIndices)):
			return defaultTensor
		columnLocalMap = self.columnFeatureLocalIndices[columnIndex]
		if not columnLocalMap:
			return defaultTensor
		defaultCPU = defaultTensor.detach().cpu()
		globalCPU = globalTensor.detach().cpu()
		defaultList = defaultCPU.tolist()
		globalList = globalCPU.tolist()
		newList = []
		for defaultValue, globalValue in zip(defaultList, globalList):
			candidates = columnLocalMap.get(int(globalValue))
			if(candidates and len(candidates) > 0):
				newList.append(int(candidates[0]))
			else:
				newList.append(int(defaultValue))
		return pt.tensor(newList, dtype=defaultTensor.dtype, device=defaultTensor.device)
	
	def populateArrays(self, tokens, sequenceObservedColumnsDict):
		#print("\n\n\n\n\npopulate_arrays:")
		
		# Optimized code for collecting indices and data for feature neurons
		cIdxList = []
		fIdxList = []
		featureListIndices = []
		featureListValues = []

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()
			numFeatures = len(fIdxTensor)

			cIdxList.append(pt.full((numFeatures,), cIdx, dtype=pt.long))
			fIdxList.append(fIdxTensor)

			if lowMem:
				featureNeurons = observedColumn.featureNeurons.coalesce()
			else:
				# Slice the globalFeatureNeurons as before
				featureNeurons = GIAANNproto_sparseTensors.sliceSparseTensor(self.databaseNetworkObject.globalFeatureNeurons, 2, observedColumn.conceptIndex)

			if (useGPUdense and not useGPUsparse):
				featureNeurons = featureNeurons.to(deviceDense)

			indices = featureNeurons.indices()  # [3, n_entries] for a 3D sparse tensor: (property, type, feature_idx)
			values = featureNeurons.values()

			# Ensure that featureIndicesInObserved is sorted if not already
			featureIndicesInObservedSorted, fIdxSortIdx = pt.sort(featureIndicesInObserved)
			fIdxTensorSorted = fIdxTensor[fIdxSortIdx]

			# Instead of expanding and comparing, directly check membership
			mask = pt.isin(indices[2], featureIndicesInObservedSorted)

			# Filter indices and values by mask
			filteredIndices = indices[:, mask]
			filteredValues = values[mask]

			if filteredIndices.size(1) > 0:
				# We need to find the corresponding f_idx for each filtered feature_idx.
				# Use searchsorted on the sorted featureIndicesInObserved
				positions = pt.searchsorted(featureIndicesInObservedSorted, filteredIndices[2])
				filteredFIdxTensor = fIdxTensorSorted[positions]
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
				filteredIndices[2] = filteredFIdxTensor
			# Insert cIdx row
			cIdxCol = pt.full((1, filteredIndices.size(1)), cIdx, dtype=pt.long, device=filteredIndices.device)
			filteredIndices = pt.cat([filteredIndices[0:2], cIdxCol, filteredIndices[2:3]], dim=0)

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
			self.featureNeuronsOriginal = self.featureNeurons.clone()

		# Now handle connections
		connectionIndicesList = []
		connectionValuesList = []

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()

			# Get indices and values from the sparse tensor
			featureConnections = observedColumn.featureConnections.coalesce()
			if not useGPUsparse:
				featureConnections = featureConnections.to(deviceDense)

			indices = featureConnections.indices()  # shape [5, n_entries]
			values = featureConnections.values()	# shape [n_entries]

			# Sort featureIndicesInObserved and fIdxTensor together if not already sorted
			featureIndicesInObservedSorted, fIdxSortIdx = pt.sort(featureIndicesInObserved)
			fIdxTensorSorted = fIdxTensor[fIdxSortIdx]

			# For each other column
			for otherCIdx, otherObservedColumn in sequenceObservedColumnsDict.items():
				otherFeatureIndicesInObserved, otherFIdxTensor = self.getObservedColumnFeatureIndices()
				otherConceptIndex = otherObservedColumn.conceptIndex

				# Sort otherFeatureIndicesInObserved and otherFIdxTensor if not sorted
				otherFeatureIndicesInObservedSorted, otherFIdxSortIdx = pt.sort(otherFeatureIndicesInObserved)
				otherFIdxTensorSorted = otherFIdxTensor[otherFIdxSortIdx]

				# Create boolean masks directly:
				maskConcept = (indices[3] == otherConceptIndex)
				maskF2 = pt.isin(indices[2], featureIndicesInObservedSorted)
				maskF4 = pt.isin(indices[4], otherFeatureIndicesInObservedSorted)

				combinedMask = maskConcept & maskF2 & maskF4

				# Filter indices and values
				filteredIndices = indices[:, combinedMask]
				filteredValues = values[combinedMask]

				# If we got no matches, filteredIndices and filteredValues will be empty.
				# We do NOT continue here; we proceed to create and append empty results as per the original requirement.

				if filteredIndices.numel() > 0:
					sourceGlobalIndices = filteredIndices[2].clone()
					targetGlobalIndices = filteredIndices[4].clone()
					# Map indices[2] back to fIdxTensor
					fIdxPositions = pt.searchsorted(featureIndicesInObservedSorted, filteredIndices[2])
					mappedFIdx = fIdxTensorSorted[fIdxPositions]

					# Map indices[4] back to otherFIdxTensor
					otherFIdxPositions = pt.searchsorted(otherFeatureIndicesInObservedSorted, filteredIndices[4])
					mappedOtherFIdx = otherFIdxTensorSorted[otherFIdxPositions]

					mappedFIdx = self.mapGlobalToLocalIndices(mappedFIdx, sourceGlobalIndices, cIdx)
					mappedOtherFIdx = self.mapGlobalToLocalIndices(mappedOtherFIdx, targetGlobalIndices, otherCIdx)

					# Adjust indices:
					# After filtering, we have:
					#   filteredIndices = [property, type, feature_idx, concept_idx, other_feature_idx]
					# We want to replace concept_idx with otherCIdx and feature_idx with mappedFIdx, other_feature_idx with mappedOtherFIdx.
					filteredIndices[2] = mappedFIdx
					filteredIndices[3] = otherCIdx
					filteredIndices[4] = mappedOtherFIdx
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

				# Insert cIdx at dimension 3 as per the original code.
				cIdxCol = pt.full((1, filteredIndices.size(1)), cIdx, dtype=pt.long, device=filteredIndices.device)
				filteredIndices = pt.cat([filteredIndices[0:2], cIdxCol, filteredIndices[2:]], dim=0)

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
			self.featureConnectionsOriginal = self.featureConnections.clone()
			if(debugDrawNeuronActivations):
				strengthSum = self.featureConnections[arrayIndexPropertiesStrengthIndex].sum().item()
	
	def updateObservedColumnsWrapper(self):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
			self.updateObservedColumns(self.sequenceObservedColumnsDict, mode="default")
		else:
			self.updateObservedColumns(self.observedColumnsDict2, mode="default")
			
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

	def flattenSparseIndices(self, indices, size):
		device = indices.device
		sizeTensor = pt.tensor(size, dtype=pt.long, device=device)
		strides = pt.ones((len(size),), dtype=pt.long, device=device)
		for i in range(len(size)-2, -1, -1):
			strides[i] = strides[i+1] * sizeTensor[i+1]
		linear = (indices * strides.unsqueeze(1)).sum(dim=0)
		return linear

	def applySparseMinUpdate(self, targetSparse, updateSparse):
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
				updatedTargetValues[matchedTargetIndices] = pt.minimum(updatedTargetValues[matchedTargetIndices], updateValues[matchMask])

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
		mask = (indices[2] == cIdx)
		if(propertyMaskLookup is not None):
			mask = mask & propertyMaskLookup[indices[0]]
		if(sequenceFeatureMaskLookup is not None):
			mask = mask & sequenceFeatureMaskLookup[indices[3]]
		filteredIndices = indices[:, mask]
		filteredValues = values[mask]
		if(filteredIndices.numel() > 0):
			filteredIndices[2] = filteredIndices[3]
			filteredIndices = filteredIndices[0:3]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				filteredIndices[2] = featureIndicesInObserved[filteredIndices[2]]
			if(insertConceptIndex is not None):
				conceptIndexRow = pt.full((1, filteredIndices.size(1)), insertConceptIndex, dtype=pt.long, device=filteredIndices.device)
				filteredIndices = pt.cat([filteredIndices[0:2], conceptIndexRow, filteredIndices[2:3]], dim=0)
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
		mask = (indices[2] == cIdx)
		if(propertyMaskLookup is not None):
			mask = mask & propertyMaskLookup[indices[0]]
		if(sequenceFeatureMaskLookup is not None):
			mask = mask & sequenceFeatureMaskLookup[indices[3]]
			mask = mask & sequenceFeatureMaskLookup[indices[5]]
		filteredIndices = indices[:, mask]
		filteredValues = values[mask]
		if(filteredIndices.numel() > 0):
			filteredIndices = pt.stack((
				filteredIndices[0],
				filteredIndices[1],
				filteredIndices[3],
				filteredIndices[4],
				filteredIndices[5]
			), dim=0)
			conceptIndicesTensor = self.conceptIndicesInSequenceObservedTensor.to(filteredIndices.device)
			filteredIndices[3] = conceptIndicesTensor[filteredIndices[3]]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				filteredIndices[2] = featureIndicesInObserved[filteredIndices[2]]
				filteredIndices[4] = featureIndicesInObserved[filteredIndices[4]]
		else:
			filteredIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=featureConnectionsSparse.device)
			filteredValues = pt.empty((0,), dtype=arrayType, device=featureConnectionsSparse.device)
		if not useGPUsparse:
			filteredIndices = filteredIndices.to(deviceSparse)
			filteredValues = filteredValues.to(deviceSparse)
		updateSparse = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=targetSize, dtype=arrayType, device=deviceSparse)
		return updateSparse

	def updateObservedColumns(self, sequenceObservedColumnsDict, mode):
		# Update observed columns with data from sequence arrays

		featureNeuronsDelta = self.featureNeurons - self.featureNeuronsOriginal
		featureConnectionsDelta = self.featureConnections - self.featureConnectionsOriginal

		featureNeuronsDeltaSparse = featureNeuronsDelta.to_sparse()
		featureConnectionsDeltaSparse = featureConnectionsDelta.to_sparse()
		featureNeuronsCurrentSparse = None
		featureConnectionsCurrentSparse = None
		replacePropertiesEnabled = arrayIndexPropertiesActivationCreate or arrayIndexPropertiesTime or arrayIndexPropertiesPos
		addPropertiesEnabled = arrayIndexPropertiesStrength or arrayIndexPropertiesPermanence
		if(replacePropertiesEnabled):
			featureNeuronsCurrentSparse = self.featureNeurons.to_sparse()
			featureConnectionsCurrentSparse = self.featureConnections.to_sparse()
		elif(arrayIndexPropertiesMinWordDistance):
			featureConnectionsCurrentSparse = self.featureConnections.to_sparse()
		if(performRedundantCoalesce):
			featureNeuronsDeltaSparse = featureNeuronsDeltaSparse.coalesce()
			featureConnectionsDeltaSparse = featureConnectionsDeltaSparse.coalesce()
			if(featureNeuronsCurrentSparse is not None):
				featureNeuronsCurrentSparse = featureNeuronsCurrentSparse.coalesce()
			if(featureConnectionsCurrentSparse is not None):
				featureConnectionsCurrentSparse = featureConnectionsCurrentSparse.coalesce()
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
				addPropertyIndicesList.append(arrayIndexPropertiesStrengthIndex)
			if(arrayIndexPropertiesPermanence):
				addPropertyIndicesList.append(arrayIndexPropertiesPermanenceIndex)
			addPropertyIndices = pt.tensor(addPropertyIndicesList, dtype=pt.long)
			addPropertyMaskLookup = self.buildMaskLookup(arrayNumberOfProperties, addPropertyIndices.to(featureNeuronsDeltaSparse.device), featureNeuronsDeltaSparse.device)
		if(replacePropertiesEnabled):
			replacePropertyMaskLookup = None
			replacePropertyIndicesList = []
			if(arrayIndexPropertiesActivationCreate):
				replacePropertyIndicesList.append(arrayIndexPropertiesActivationIndex)
			if(arrayIndexPropertiesTime):
				replacePropertyIndicesList.append(arrayIndexPropertiesTimeIndex)
			if(arrayIndexPropertiesPos):
				replacePropertyIndicesList.append(arrayIndexPropertiesPosIndex)
			replacePropertyIndices = pt.tensor(replacePropertyIndicesList, dtype=pt.long)
			replacePropertyMaskLookup = self.buildMaskLookup(arrayNumberOfProperties, replacePropertyIndices.to(featureNeuronsDeltaSparse.device), featureNeuronsDeltaSparse.device)
		if(arrayIndexPropertiesMinWordDistance):
			minPropertyIndices = pt.tensor([arrayIndexPropertiesMinWordDistanceIndex], dtype=pt.long)
			minPropertyMaskLookup = self.buildMaskLookup(arrayNumberOfProperties, minPropertyIndices.to(featureConnectionsCurrentSparse.device), featureConnectionsCurrentSparse.device)

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

		if not lowMem:
			globalFeatureNeurons = self.databaseNetworkObject.globalFeatureNeurons.coalesce()

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			conceptIndex = observedColumn.conceptIndex

			if lowMem:
				featureTargetSparse = observedColumn.featureNeurons.coalesce()
				featureTargetSize = featureTargetSparse.size()
			else:
				featureTargetSparse = globalFeatureNeurons
				featureTargetSize = featureTargetSparse.size()

			if(addPropertiesEnabled):
				featureUpdatesAdd = self.extractSequenceFeatureUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureNeuronsDeltaSparse, addPropertyMaskLookup, sequenceFeatureMaskLookup, featureTargetSize, insertConceptIndex=None if lowMem else conceptIndex)
			if(replacePropertiesEnabled):
				featureUpdatesReplace = self.extractSequenceFeatureUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureNeuronsCurrentSparse, replacePropertyMaskLookup, sequenceFeatureMaskLookup, featureTargetSize, insertConceptIndex=None if lowMem else conceptIndex)

				featureTargetSparse = featureTargetSparse.coalesce()
				targetIndices = featureTargetSparse.indices()
				targetValues = featureTargetSparse.values()
				if lowMem:
					removeMask = replacePropertyMaskLookup[targetIndices[0]] & observedFeatureMaskLookup[targetIndices[2]]
				else:
					removeMask = replacePropertyMaskLookup[targetIndices[0]] & (targetIndices[2] == conceptIndex) & observedFeatureMaskLookup[targetIndices[3]]
				keepMask = pt.logical_not(removeMask)
				filteredTargetIndices = targetIndices[:, keepMask]
				filteredTargetValues = targetValues[keepMask]
				featureTargetSparse = pt.sparse_coo_tensor(filteredTargetIndices, filteredTargetValues, size=featureTargetSize, dtype=arrayType, device=deviceSparse)

				combinedFeatureUpdates = self.combineSparseUpdates(featureUpdatesAdd, featureUpdatesReplace, featureTargetSize)
			else:
				combinedFeatureUpdates = featureUpdatesAdd
			featureTargetSparse = self.addSparseUpdate(featureTargetSparse, combinedFeatureUpdates)

			if lowMem:
				observedColumn.featureNeurons = featureTargetSparse
			else:
				globalFeatureNeurons = featureTargetSparse

			observedColumn.featureConnections = observedColumn.featureConnections.coalesce()
			connectionTargetSize = observedColumn.featureConnections.size()
			if(addPropertiesEnabled):
				connectionUpdatesAdd = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsDeltaSparse, addPropertyMaskLookup, sequenceFeatureMaskLookup, connectionTargetSize)
			if(replacePropertiesEnabled):
				connectionUpdatesReplace = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsCurrentSparse, replacePropertyMaskLookup, sequenceFeatureMaskLookup, connectionTargetSize)
			if(arrayIndexPropertiesMinWordDistance):
				connectionUpdatesMin = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsCurrentSparse, minPropertyMaskLookup, sequenceFeatureMaskLookup, connectionTargetSize)
			if(replacePropertiesEnabled):
				connectionIndices = observedColumn.featureConnections.indices()
				connectionValues = observedColumn.featureConnections.values()
				removeMaskConnections = replacePropertyMaskLookup[connectionIndices[0]]
				removeMaskConnections = removeMaskConnections & observedFeatureMaskLookup[connectionIndices[2]]
				removeMaskConnections = removeMaskConnections & sequenceConceptMaskLookup[connectionIndices[3]]
				removeMaskConnections = removeMaskConnections & observedFeatureMaskLookup[connectionIndices[4]]
				keepConnectionsMask = pt.logical_not(removeMaskConnections)
				filteredConnectionIndices = connectionIndices[:, keepConnectionsMask]
				filteredConnectionValues = connectionValues[keepConnectionsMask]
				observedColumn.featureConnections = pt.sparse_coo_tensor(filteredConnectionIndices, filteredConnectionValues, size=connectionTargetSize, dtype=arrayType, device=deviceSparse)

				combinedConnectionUpdates = self.combineSparseUpdates(connectionUpdatesAdd, connectionUpdatesReplace, connectionTargetSize)
			else:
				combinedConnectionUpdates = connectionUpdatesAdd
			if(addPropertiesEnabled):
				observedColumn.featureConnections = self.addSparseUpdate(observedColumn.featureConnections, combinedConnectionUpdates)
			if(arrayIndexPropertiesMinWordDistance):
				observedColumn.featureConnections = self.applySparseMinUpdate(observedColumn.featureConnections, connectionUpdatesMin)

		if not lowMem:
			self.databaseNetworkObject.globalFeatureNeurons = globalFeatureNeurons
