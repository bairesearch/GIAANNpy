"""GIAANNproto_sequenceObservedColumnsExcitation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

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
	def __init__(self, databaseNetworkObject, tokens, observedColumnsDict, observedColumnsSequenceWordIndexDict, inferenceMode):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.databaseNetworkObject = databaseNetworkObject
		self.observedColumnsDict = observedColumnsDict	# key: lemma, value: ObservedColumn
		self.observedColumnsSequenceWordIndexDict = observedColumnsSequenceWordIndexDict	# key: sequence word index, value: ObservedColumn
		self.noDelimiterDetectedBetweenConceptTokens = False

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
			self.featureConnections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

			self.featureNeuronsOriginal = self.featureNeurons.clone()
			self.featureConnectionsOriginal = self.featureConnections.clone()

			# Populate arrays with data from observedColumnsDict (required for inference)
			if(useInference):
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
			self.featureNeuronsOriginal = None
			self.featureConnectionsOriginal = None

			
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
	
	def removeDuplicates(self, lst):
		#python requires ordered sets
		lst = list(dict.fromkeys(lst))
		return lst
				
	@staticmethod
	def initialiseFeatureNeuronsSequence(cs, fs):
		featureNeurons = pt.zeros(arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, dtype=arrayType)
		return featureNeurons

	@staticmethod
	def initialiseFeatureConnectionsSequence(cs, fs):
		featureConnections = pt.zeros(arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs, dtype=arrayType)
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
			self.featureNeuronsOriginal = self.featureNeurons.clone()

		# Now handle connections
		connectionIndicesList = []
		connectionValuesList = []

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()
			if(useGPUsparseStrict and not useGPUsparse):
				featureIndicesInObserved = featureIndicesInObserved.to(deviceSparse)
				fIdxTensor = fIdxTensor.to(deviceSparse)

			# Get indices and values from the sparse tensor
			featureConnections = observedColumn.featureConnections.coalesce()
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
			self.featureConnectionsOriginal = self.featureConnections.clone()
			if(debugDrawNeuronActivations):
				strengthSum = self.featureConnections[arrayIndexPropertiesStrengthIndex].sum().item()
	
	def updateObservedColumnsWrapper(self, inference=False):
		self.debugInferenceActive = inference
		self.inferenceConceptUpdateCounts = {} if inference else None
		if(trainSequenceObservedColumnsMatchSequenceWords):
			#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
			self.updateObservedColumns(self.sequenceObservedColumnsDict, inference, mode="default")
		else:
			self.updateObservedColumns(self.observedColumnsDict2, inference, mode="default")

	def debugPrintPersistedColumnSummary(self, mode):
		debugEnabled = debugPrintSequenceObservedColumnsConnections
		if(debugEnabled):
			targetLemma = "movement"
			observedColumn = self.observedColumnsDict.get(targetLemma)
			if(observedColumn is None):
				print(f"debugPersistedColumn ({mode}): lemma '{targetLemma}' not in observedColumnsDict")
			else:
				connectionsStrength = observedColumn.featureConnections[arrayIndexPropertiesStrengthIndex]
				if(connectionsStrength.is_sparse):
					connectionsStrength = connectionsStrength.to_dense()
				outgoingCount = 0
				maxStrength = 0.0
				internalCount = 0
				externalCount = 0
				segmentCounts = None
				lastSegmentCount = 0
				targetTop = []
				if(connectionsStrength.numel() > 0):
					outgoingMask = connectionsStrength > 0
					outgoingCount = int(outgoingMask.sum().item())
					maxStrength = float(connectionsStrength.max().item())
					internalMask = outgoingMask[:, :, :, observedColumn.conceptIndex, :]
					internalCount = int(internalMask.sum().item())
					externalCount = outgoingCount - internalCount
					segmentCounts = outgoingMask.sum(dim=(0, 2, 3, 4)).to("cpu").tolist()
					lastSegmentCount = int(outgoingMask[:, arrayIndexSegmentLast].sum().item())
					targetCounts = outgoingMask.sum(dim=(0, 1, 2, 4))
					if(targetCounts.numel() > 0):
						topkCount = min(3, int(targetCounts.shape[0]))
						topkValues, topkIndices = pt.topk(targetCounts, topkCount)
						targetTop = [(int(idx.item()), int(val.item())) for idx, val in zip(topkIndices, topkValues) if int(val.item()) > 0]
				print(f"debugPersistedColumn ({mode}): lemma={targetLemma}, outgoing>0={outgoingCount}, maxStrength={maxStrength}, internal>0={internalCount}, external>0={externalCount}, lastSegment>0={lastSegmentCount}")
				if(segmentCounts is not None):
					print(f"\tsegmentCounts>0={segmentCounts}")
				if(len(targetTop) > 0):
					targetLabels = []
					for targetIndex, targetCount in targetTop:
						targetLemma = "<unknown>"
						if(targetIndex < len(observedColumn.databaseNetworkObject.conceptColumnsList)):
							targetLemma = observedColumn.databaseNetworkObject.conceptColumnsList[targetIndex]
						targetLabels.append(f"{targetLemma}:{targetCount}")
					print(f"\ttopTargets={targetLabels}")
		return
			
	def updateObservedColumns(self, sequenceObservedColumnsDict, inference, mode):
		if(arrayIndexPropertiesEfficient and not inference):
			self.updateObservedColumnsEfficient(sequenceObservedColumnsDict, mode)
		else:
			self.updateObservedColumnsVerbose(sequenceObservedColumnsDict, mode)
	
	def updateObservedColumnsVerbose(self, sequenceObservedColumnsDict, mode):
		# Update observed columns with data from sequence arrays

		inferenceConceptUpdateCounts = self.inferenceConceptUpdateCounts if getattr(self, "debugInferenceActive", False) else None
		featureNeuronsDelta = self.featureNeurons - self.featureNeuronsOriginal
		featureConnectionsDelta = self.featureConnections - self.featureConnectionsOriginal
		if(useGPUsparseStrict and not useGPUsparse):
			featureNeuronsDelta = featureNeuronsDelta.to(deviceSparse)
			featureConnectionsDelta = featureConnectionsDelta.to(deviceSparse)

		featureNeuronsDeltaSparse = featureNeuronsDelta.to_sparse()
		featureConnectionsDeltaSparse = featureConnectionsDelta.to_sparse()
		featureNeuronsCurrentSparse = None
		featureConnectionsCurrentSparse = None

		replacePropertiesEnabled = arrayIndexPropertiesActivation or arrayIndexPropertiesTime	
		assert not replacePropertiesEnabled, "replacePropertiesEnabled is not robust to duplicate features"
		addPropertiesEnabled = arrayIndexPropertiesStrength or arrayIndexPropertiesPermanence
		
		if(replacePropertiesEnabled):
			featureNeuronsCurrentSparse = self.featureNeurons.to_sparse()
			featureConnectionsCurrentSparse = self.featureConnections.to_sparse()
		elif(arrayIndexPropertiesMinWordDistance):
			featureConnectionsCurrentSparse = self.featureConnections.to_sparse()
		elif(arrayIndexPropertiesPos):
			featureNeuronsCurrentSparse = self.featureNeurons.to_sparse()
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
				addPropertyIndicesList.append(arrayIndexPropertiesStrengthIndex)
			if(arrayIndexPropertiesPermanence):
				addPropertyIndicesList.append(arrayIndexPropertiesPermanenceIndex)
			addPropertyIndices = pt.tensor(addPropertyIndicesList, dtype=pt.long)
			addPropertyMaskLookup = self.buildMaskLookup(arrayNumberOfProperties, addPropertyIndices.to(featureNeuronsDeltaSparse.device), featureNeuronsDeltaSparse.device)
		posPropertyMaskLookupFeature = None
		posPropertyMaskLookupConnection = None
		posPropertyIndices = None
		if(arrayIndexPropertiesPos):
			posPropertyIndices = pt.tensor([arrayIndexPropertiesPosIndex], dtype=pt.long)
			posPropertyMaskLookupFeature = self.buildMaskLookup(arrayNumberOfProperties, posPropertyIndices.to(featureNeuronsDeltaSparse.device), featureNeuronsDeltaSparse.device)
		if(replacePropertiesEnabled):
			replacePropertyMaskLookup = None
			replacePropertyIndicesList = []
			if(arrayIndexPropertiesActivationCreate):
				replacePropertyIndicesList.append(arrayIndexPropertiesActivationIndex)
			if(arrayIndexPropertiesTime):
				replacePropertyIndicesList.append(arrayIndexPropertiesTimeIndex)
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
		if(arrayIndexPropertiesPos):
			posPropertyMaskLookupConnection = self.buildMaskLookup(arrayNumberOfProperties, posPropertyIndices.to(connectionDevice), connectionDevice)

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
				if(self.debugInferenceActive and multipleDendriticBranches and updateCount is not None and updateCount > 1 and arrayIndexPropertiesActivationIndex is not None):
					preserveActivationOnReplace = True
				if(self.debugInferenceActive and multipleDendriticBranches and arrayIndexPropertiesActivationIndex is not None):
					activationUpdates = featureUpdatesReplace.coalesce()
					activationUpdateIndices = activationUpdates.indices()
					activationUpdateMask = (activationUpdateIndices[0] == arrayIndexPropertiesActivationIndex)
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
					removeMask = removeMask & (targetIndices[0] != arrayIndexPropertiesActivationIndex)
				elif(self.debugInferenceActive and multipleDendriticBranches and arrayIndexPropertiesActivationIndex is not None):
					if(activationUpdateBranches is not None and activationUpdateBranches.numel() > 0):
						activationBranchLookup = self.buildMaskLookup(numberOfDendriticBranches, activationUpdateBranches.to(targetIndices.device), targetIndices.device)
						activationMask = (targetIndices[0] == arrayIndexPropertiesActivationIndex)
						if(lowMem):
							activationMask = activationMask & observedFeatureMaskLookup[targetIndices[3]]
						else:
							activationMask = activationMask & (targetIndices[3] == conceptIndex) & observedFeatureMaskLookup[targetIndices[4]]
						activationBranchMask = activationMask & activationBranchLookup[targetIndices[1]]
						removeMask = (removeMask & pt.logical_not(activationMask)) | activationBranchMask
					else:
						removeMask = removeMask & (targetIndices[0] != arrayIndexPropertiesActivationIndex)
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

			observedColumn.featureConnections = observedColumn.featureConnections.coalesce()
			connectionTargetSize = observedColumn.featureConnections.size()
			if(addPropertiesEnabled):
				connectionUpdatesAdd = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsDeltaSparse, addPropertyMaskLookup, sequenceFeatureMaskLookup, connectionTargetSize)
			if(replacePropertiesEnabled):
				connectionUpdatesReplace = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsCurrentSparse, replacePropertyMaskLookup, sequenceFeatureMaskLookup, connectionTargetSize)
			if(arrayIndexPropertiesPos):
				connectionUpdatesPos = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsCurrentSparse, posPropertyMaskLookupConnection, sequenceFeatureMaskLookup, connectionTargetSize)
			if(arrayIndexPropertiesMinWordDistance):
				connectionUpdatesMin = self.extractSequenceConnectionUpdates(cIdx, fIdxTensor, featureIndicesObservedDevice, featureConnectionsCurrentSparse, minPropertyMaskLookup, sequenceFeatureMaskLookup, connectionTargetSize)
			if(replacePropertiesEnabled):
				connectionIndices = observedColumn.featureConnections.indices()
				connectionValues = observedColumn.featureConnections.values()
				removeMaskConnections = replacePropertyMaskLookup[connectionIndices[0]]
				removeMaskConnections = removeMaskConnections & observedFeatureMaskLookup[connectionIndices[3]]
				removeMaskConnections = removeMaskConnections & sequenceConceptMaskLookup[connectionIndices[4]]
				removeMaskConnections = removeMaskConnections & observedFeatureMaskLookup[connectionIndices[5]]
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
			if(arrayIndexPropertiesPos):
				observedColumn.featureConnections = self.applySparseMaxUpdate(observedColumn.featureConnections, connectionUpdatesPos)

		if not lowMem:
			self.databaseNetworkObject.globalFeatureNeurons = globalFeatureNeurons

	def updateObservedColumnsEfficient(self, sequenceObservedColumnsDict, mode):
		if not arrayIndexPropertiesStrength:
			return

		featureNeuronsDelta = self.featureNeurons[arrayIndexPropertiesStrengthIndex] - self.featureNeuronsOriginal[arrayIndexPropertiesStrengthIndex]
		featureConnectionsDelta = self.featureConnections[arrayIndexPropertiesStrengthIndex] - self.featureConnectionsOriginal[arrayIndexPropertiesStrengthIndex]
		if(useGPUsparseStrict and not useGPUsparse):
			featureNeuronsDelta = featureNeuronsDelta.to(deviceSparse)
			featureConnectionsDelta = featureConnectionsDelta.to(deviceSparse)

		featureNeuronsDeltaSparse = featureNeuronsDelta.to_sparse()
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
		featureRanges, featureIndicesSorted, featureValuesSorted = self.buildSparseColumnRanges(featureIndices, featureValues, 2)

		connectionIndices = featureConnectionsDeltaSparse.indices()
		connectionValues = featureConnectionsDeltaSparse.values()
		connectionIndices, connectionValues = self.filterSparseByFeatureMask(connectionIndices, connectionValues, sequenceFeatureMaskConnection, [3, 5])
		connectionRanges, connectionIndicesSorted, connectionValuesSorted = self.buildSparseColumnRanges(connectionIndices, connectionValues, 2)

		connectionMinRanges = {}
		connectionMinIndicesSorted = None
		connectionMinValuesSorted = None
		if(arrayIndexPropertiesMinWordDistance):
			connectionMin = self.featureConnections[arrayIndexPropertiesMinWordDistanceIndex]
			connectionMinSparse = connectionMin.to_sparse()
			if not useGPUsparse:
				connectionMinSparse = connectionMinSparse.to(deviceSparse)
			connectionMinIndices = connectionMinSparse.indices()
			connectionMinValues = connectionMinSparse.values()
			connectionMinIndices, connectionMinValues = self.filterSparseByFeatureMask(connectionMinIndices, connectionMinValues, sequenceFeatureMaskConnection, [3, 5])
			connectionMinRanges, connectionMinIndicesSorted, connectionMinValuesSorted = self.buildSparseColumnRanges(connectionMinIndices, connectionMinValues, 2)

		conceptIndicesTensor = self.conceptIndicesInSequenceObservedTensor.to(connectionDevice)

		if not lowMem:
			globalFeatureNeurons = self.databaseNetworkObject.globalFeatureNeurons
			if(combineSparseUpdatesPerSequence):
				globalFeatureNeuronUpdates = []

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			conceptIndex = observedColumn.conceptIndex

			if lowMem:
				featureTargetSparse = observedColumn.featureNeurons
				featureTargetSize = featureTargetSparse.size()
			else:
				if(combineSparseUpdatesPerSequence):
					featureTargetSparse = None
				else:
					featureTargetSparse = globalFeatureNeurons
				featureTargetSize = globalFeatureNeurons.size()

			featureRange = featureRanges.get(cIdx)
			if(featureRange is not None):
				start, end = featureRange
				featureUpdateIndices = featureIndicesSorted[:, start:end]
				featureUpdateValues = featureValuesSorted[start:end]
				featureUpdates = self.buildFeaturePropertyUpdateSparse(featureUpdateIndices, featureUpdateValues, arrayIndexPropertiesStrengthIndex, featureIndicesObservedFeatureDevice, featureTargetSize, insertConceptIndex=None if lowMem else conceptIndex)
				if lowMem:
					featureTargetSparse = self.addSparseUpdateNonNegative(featureTargetSparse, featureUpdates)
				else:
					if combineSparseUpdatesPerSequence:
						globalFeatureNeuronUpdates.append(featureUpdates)
					else:
						featureTargetSparse = self.addSparseUpdateNonNegative(featureTargetSparse, featureUpdates)

			if lowMem:
				observedColumn.featureNeurons = featureTargetSparse
			else:
				if not combineSparseUpdatesPerSequence:
					globalFeatureNeurons = featureTargetSparse

			connectionTargetSparse = observedColumn.featureConnections
			connectionTargetSize = connectionTargetSparse.size()

			connectionRange = connectionRanges.get(cIdx)
			if(connectionRange is not None):
				start, end = connectionRange
				connectionUpdateIndices = connectionIndicesSorted[:, start:end]
				connectionUpdateValues = connectionValuesSorted[start:end]
				connectionUpdates = self.buildConnectionPropertyUpdateSparse(connectionUpdateIndices, connectionUpdateValues, arrayIndexPropertiesStrengthIndex, featureIndicesObservedConnectionDevice, conceptIndicesTensor, connectionTargetSize)
				connectionTargetSparse = self.addSparseUpdateNonNegative(connectionTargetSparse, connectionUpdates)

			if(arrayIndexPropertiesMinWordDistance):
				minRange = connectionMinRanges.get(cIdx)
				if(minRange is not None):
					start, end = minRange
					minUpdateIndices = connectionMinIndicesSorted[:, start:end]
					minUpdateValues = connectionMinValuesSorted[start:end]
					minUpdates = self.buildConnectionPropertyUpdateSparse(minUpdateIndices, minUpdateValues, arrayIndexPropertiesMinWordDistanceIndex, featureIndicesObservedConnectionDevice, conceptIndicesTensor, connectionTargetSize)
					connectionTargetSparse = self.applySparseMinUpdate(connectionTargetSparse, minUpdates)

			observedColumn.featureConnections = connectionTargetSparse

		if not lowMem:
			if(combineSparseUpdatesPerSequence):
				if len(globalFeatureNeuronUpdates) > 0:
					combinedFeatureUpdates = self.combineSparseUpdatesList(globalFeatureNeuronUpdates, globalFeatureNeurons.size())
					globalFeatureNeurons = self.addSparseUpdateNonNegative(globalFeatureNeurons, combinedFeatureUpdates)
			self.databaseNetworkObject.globalFeatureNeurons = globalFeatureNeurons

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
	
	if(combineSparseUpdatesPerSequence):
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

	def addSparseUpdateNonNegative(self, targetSparse, updateSparse):
		if(updateSparse._nnz() == 0):
			return targetSparse
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
		return combinedSparse.coalesce()

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
