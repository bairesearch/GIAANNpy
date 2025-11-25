"""GIAANNproto_databaseNetworkTrain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Train

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors


# Define the SequenceObservedColumns class
class SequenceObservedColumns:
	"""
	Contains sequence observed columns object arrays which stack a feature subset of the observed columns object arrays for the current sequence.
	"""
	def __init__(self, databaseNetworkObject, words, lemmas, observedColumnsDict, observedColumnsSequenceWordIndexDict):
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
		self.words = words
		self.lemmas = lemmas
		#identify feature indices from complete ObservedColumns.featureNeurons or globalFeatureNeurons feature lists currently stored in SequenceObservedColumns.featureNeurons	#required for useInference
		observedColumn = list(observedColumnsDict.values())[0]	#all features (including words) are identical per observed column
		self.featureWords, self.featureIndicesInObservedTensor, self.fIdxTensor = self.identifyObservedColumnFeatureWords(words, lemmas, observedColumn)

		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
			self.fs = self.featureIndicesInObservedTensor.shape[0]
		else:
			self.fs = len(self.featureWords)
		self.featureWordToIndex = {}
		self.indexToFeatureWord = {}
		for idx, featureWord in enumerate(self.featureWords):
			self.featureWordToIndex[featureWord] = idx
			self.indexToFeatureWord[idx] = featureWord

		# Initialize arrays
		self.featureNeurons = self.initialiseFeatureNeuronsSequence(self.cs, self.fs)
		self.featureConnections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

		# Populate arrays with data from observedColumnsDict
		if(trainSequenceObservedColumnsMatchSequenceWords):
			self.populateArrays(words, lemmas, self.sequenceObservedColumnsDict)
		else:
			self.populateArrays(words, lemmas, self.observedColumnsDict2)

			
	def identifyObservedColumnFeatureWords(self, words, lemmas, observedColumn):
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
			featureWords = []
			featureIndicesInObserved = []
			#print("\nidentifyObservedColumnFeatureWords: words = ", len(words))
			for wordIndex, (word, lemma) in enumerate(zip(words, lemmas)):
				featureWord = word
				featureLemma = lemma
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
	
	def populateArrays(self, words, lemmas, sequenceObservedColumnsDict):
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
					# Map indices[2] back to fIdxTensor
					fIdxPositions = pt.searchsorted(featureIndicesInObservedSorted, filteredIndices[2])
					mappedFIdx = fIdxTensorSorted[fIdxPositions]

					# Map indices[4] back to otherFIdxTensor
					otherFIdxPositions = pt.searchsorted(otherFeatureIndicesInObservedSorted, filteredIndices[4])
					mappedOtherFIdx = otherFIdxTensorSorted[otherFIdxPositions]

					# Adjust indices:
					# After filtering, we have:
					#   filteredIndices = [property, type, feature_idx, concept_idx, other_feature_idx]
					# We want to replace concept_idx with otherCIdx and feature_idx with mappedFIdx, other_feature_idx with mappedOtherFIdx.
					filteredIndices[2] = mappedFIdx
					filteredIndices[3] = otherCIdx
					filteredIndices[4] = mappedOtherFIdx
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

	
	def updateObservedColumnsWrapper(self):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
			self.updateObservedColumns(self.sequenceObservedColumnsDict, mode="default")
		else:
			self.updateObservedColumns(self.observedColumnsDict2, mode="default")
			
	def updateObservedColumns(self, sequenceObservedColumnsDict, mode):
		# Update observed columns with data from sequence arrays
			
		featureNeurons = self.featureNeurons - self.featureNeuronsOriginal	#convert to changes
		featureConnections = self.featureConnections - self.featureConnectionsOriginal	#convert to changes
		
		featureNeurons = featureNeurons.to_sparse()
		featureConnections = featureConnections.to_sparse()
		if(performRedundantCoalesce):
			featureNeurons = featureNeurons.coalesce()
			featureConnections = featureConnections.coalesce()		

		for cIdx, observedColumn in sequenceObservedColumnsDict.items():
			featureIndicesInObserved, fIdxTensor = self.getObservedColumnFeatureIndices()
			conceptIndex = observedColumn.conceptIndex

			if lowMem:
				observedColumn.featureNeurons = observedColumn.featureNeurons.coalesce()
			else:
				#temporarily store slices of the globalFeatureNeurons array in the observed_columns (used by update_observed_columns only)
				observedColumn.featureNeurons = GIAANNproto_sparseTensors.sliceSparseTensor(self.databaseNetworkObject.globalFeatureNeurons, 2, conceptIndex)	
			
			# feature neurons;
			indices = featureNeurons.indices()
			values = featureNeurons.values()
			mask = (indices[2] == cIdx) & pt.isin(indices[3], fIdxTensor)
			filteredIndices = indices[:, mask]
			filteredValues = values[mask]
			filteredIndices[2] = filteredIndices[3]
			filteredIndices = filteredIndices[0:3]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				filteredIndices[2] = featureIndicesInObserved[filteredIndices[2]]
			if lowMem:
				observedColumn.featureNeurons = observedColumn.featureNeurons + pt.sparse_coo_tensor(filteredIndices, filteredValues, size=observedColumn.featureNeurons.size(), dtype=arrayType, device=deviceSparse)
				observedColumn.featureNeurons = observedColumn.featureNeurons.coalesce()
				observedColumn.featureNeurons.values().clamp_(min=0)
			else:
				self.featureNeuronChanges[cIdx] = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=observedColumn.featureNeurons.size(), dtype=arrayType, device=deviceSparse)
			
			# feature connections;
			indices = featureConnections.indices()
			values = featureConnections.values()
			mask = (indices[2] == cIdx)
			filteredIndices = indices[:, mask]
			filteredValues = values[mask]
			filteredIndices[2] = filteredIndices[3]
			filteredIndices[3] = filteredIndices[4]
			filteredIndices[4] = filteredIndices[5]
			filteredIndices = filteredIndices[0:5]
			filteredIndices[3] = self.conceptIndicesInSequenceObservedTensor[filteredIndices[3]]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				filteredIndices[2] = featureIndicesInObserved[filteredIndices[2]]
				filteredIndices[4] = featureIndicesInObserved[filteredIndices[4]]
			observedColumn.featureConnections = observedColumn.featureConnections + pt.sparse_coo_tensor(filteredIndices, filteredValues, size=observedColumn.featureConnections.size(), dtype=arrayType, device=deviceSparse)
			observedColumn.featureConnections = observedColumn.featureConnections.coalesce()
			observedColumn.featureConnections.values().clamp_(min=0)
	
		if not lowMem:
			observedColumnFeatureNeuronsDict = {}
			for cIdx, observedColumn in sequenceObservedColumnsDict.items():
				conceptIndex = observedColumn.conceptIndex
				observedColumnFeatureNeuronsDict[conceptIndex] = self.featureNeuronChanges[cIdx]
			self.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.mergeTensorSlicesSum(self.databaseNetworkObject.globalFeatureNeurons, observedColumnFeatureNeuronsDict, 2)


def createConceptMask(sequenceObservedColumns, lemmas):
	conceptMask = pt.tensor([i in sequenceObservedColumns.columnsIndexSequenceWordIndexDict for i in range(len(lemmas))], dtype=pt.bool)
	conceptIndices = pt.nonzero(conceptMask).squeeze(1)
	numberConcepts = conceptIndices.shape[0]
	return conceptMask, conceptIndices, numberConcepts
	
def processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, words, lemmas, posTags, train=True, firstSeedTokenIndex=None, numSeedTokens=None):
	"""
	For every concept word (lemma) in the sequence, identify every feature neuron in that column that occurs q words before or after the concept word in the sequence, including the concept neuron. This function has been parallelized using PyTorch array operations.
	"""

	if not usePOS:
		q = 5  # Fixed window size when not using POS tags

	# Identify all concept word indices
	conceptMask, conceptIndices, numberConceptsInSequence = createConceptMask(sequenceObservedColumns, lemmas)
	#conceptIndices may be slightly longer than number of unique columns in sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
	if numberConceptsInSequence == 0:
		return  # No concept words to process

	if usePOS:
		# Sort conceptIndices
		conceptIndicesSorted = conceptIndices.sort().values
		
		# Find previous concept indices for each concept index
		prevConceptPositions = pt.searchsorted(conceptIndicesSorted, conceptIndices, right=False) - 1
		prevConceptExists = prevConceptPositions >= 0
		prevConceptPositions = prevConceptPositions.clamp(min=0)
		prevConceptIndices = pt.where(prevConceptExists, conceptIndicesSorted[prevConceptPositions], pt.zeros_like(conceptIndices))
		distToPrevConcept = pt.where(prevConceptExists, conceptIndices - prevConceptIndices, conceptIndices+1) #If no previous concept, distance is the index itself
		
		# Find next concept indices for each concept index
		nextConceptPositions = pt.searchsorted(conceptIndicesSorted, conceptIndices, right=True)
		nextConceptExists = nextConceptPositions < len(conceptIndices)
		nextConceptPositions = nextConceptPositions.clamp(max=len(nextConceptPositions)-1)
		nextConceptIndices = pt.where(nextConceptExists, conceptIndicesSorted[nextConceptPositions], pt.full_like(conceptIndices, len(sequence)))
		distToNextConcept = pt.where(nextConceptExists, nextConceptIndices - conceptIndices, len(sequence) - conceptIndices)
	else:
		q = 5
		distToPrevConcept = pt.full((conceptIndices.size(0),), q, dtype=pt.long)
		distToNextConcept = pt.full((conceptIndices.size(0),), q, dtype=pt.long)

	# Calculate start and end indices for each concept word
	if(debugConceptFeaturesOccurFirstInSubsequence):
		if usePOS:
			startIndices = (conceptIndices).clamp(min=0)
			endIndices = (conceptIndices + distToNextConcept).clamp(max=len(sequence))
		else:
			startIndices = (conceptIndices).clamp(min=0)
			endIndices = (conceptIndices + q + 1).clamp(max=len(sequence))	
	else:
		if usePOS:
			startIndices = (conceptIndices - distToPrevConcept + 1).clamp(min=0)
			endIndices = (conceptIndices + distToNextConcept).clamp(max=len(sequence))
		else:
			startIndices = (conceptIndices - q).clamp(min=0)
			endIndices = (conceptIndices + q + 1).clamp(max=len(sequence))

	processFeatures(sequenceObservedColumns, sequenceIndex, startIndices, endIndices, sequence, words, lemmas, posTags, conceptIndices, train, firstSeedTokenIndex, numSeedTokens)
	
	return conceptIndices, startIndices, endIndices

def processFeatures(sequenceObservedColumns, sequenceIndex, startIndices, endIndices, sequence, words, lemmas, posTags, conceptIndices, train, firstSeedTokenIndex=None, numSeedTokens=None):
	numberConceptsInSequence = conceptIndices.shape[0]
	
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs
	featureNeuronsActive = pt.zeros((arrayNumberOfSegments, cs, fs), dtype=arrayType)
	featureNeuronsWordOrder = pt.arange(fs).unsqueeze(0).repeat(cs, 1)
	pt.zeros((cs, fs), dtype=pt.long)
	columnsWordOrder = pt.zeros((cs), dtype=pt.long)
	featureNeuronsPos = pt.zeros((cs, fs), dtype=arrayType)
	if(trainSequenceObservedColumnsMatchSequenceWords):
		sequenceConceptIndexMask = pt.ones((cs, fs), dtype=arrayType)
	else:
		sequenceConceptIndexMask = None
	if(useSANI):
		featureNeuronsSegmentMask = pt.zeros((cs, arrayNumberOfSegments), dtype=arrayType)	#note this mask is for permanence updates (it assumes that the network has been constructed with forward column connections only)
	else:
		featureNeuronsSegmentMask = pt.ones((cs, arrayNumberOfSegments), dtype=arrayType)
	
	conceptIndicesList = conceptIndices.tolist()
	for i, sequenceConceptWordIndex in enumerate(conceptIndicesList):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			sequenceConceptIndex = i
		else:
			conceptLemma = lemmas[sequenceConceptWordIndex]
			sequenceConceptIndex = sequenceObservedColumns.conceptNameToIndex[conceptLemma] 
				
		if(useSANI):
			numberOfSegments = min(arrayNumberOfSegments, i+1)
			featureNeuronsSegmentMask[sequenceConceptIndex, :] = pt.cat([pt.zeros(arrayNumberOfSegments-numberOfSegments), pt.ones(numberOfSegments)], dim=0)
			minSequentialSegmentIndex = max(0, arrayNumberOfSegments-sequenceConceptIndex-1)
			activeSequentialSegments = pt.arange(minSequentialSegmentIndex, arrayNumberOfSegments, 1)
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly and trainSequenceObservedColumnsMatchSequenceWords):
			if(useSANI):
				featureNeuronsActive[activeSequentialSegments, sequenceConceptIndex, startIndices[sequenceConceptIndex]:endIndices[sequenceConceptIndex]] = 1
			else:
				featureNeuronsActive[arrayIndexSegmentFirst, sequenceConceptIndex, startIndices[sequenceConceptIndex]:endIndices[sequenceConceptIndex]] = 1
			columnsWordOrder[sequenceConceptIndex] = sequenceConceptIndex
			sequenceConceptIndexMask[:, sequenceConceptWordIndex] = 0
			sequenceConceptIndexMask[sequenceConceptIndex, sequenceConceptWordIndex] = 1
			for j in range(startIndices[sequenceConceptIndex], endIndices[sequenceConceptIndex]):
				featurePos = posStringToPosInt(sequenceObservedColumns.databaseNetworkObject.nlp, posTags[j])
				featureNeuronsPos[sequenceConceptIndex, j] = featurePos
				featureNeuronsWordOrder[sequenceConceptIndex, j] = j
		else:
			for j in range(startIndices[i], endIndices[i]):
				featureWord = words[j].lower()
				featureLemma = lemmas[j]
				featurePos = posStringToPosInt(sequenceObservedColumns.databaseNetworkObject.nlp, posTags[j])
				if(j in sequenceObservedColumns.columnsIndexSequenceWordIndexDict):
					sequenceConceptWordIndex = j
					columnsWordOrder[sequenceConceptIndex] = sequenceConceptIndex
					if(useDedicatedConceptNames2):
						sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[variableConceptNeuronFeatureName]
					else:
						sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[featureLemma]
					if(useSANI):
						featureNeuronsActive[activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
					else:
						featureNeuronsActive[arrayIndexSegmentFirst, sequenceConceptIndex, sequenceFeatureIndex] = 1
				elif(featureWord in sequenceObservedColumns.featureWordToIndex):
					sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[featureWord]
					if(useSANI):
						featureNeuronsActive[activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
					else:
						featureNeuronsActive[arrayIndexSegmentFirst, sequenceConceptIndex, sequenceFeatureIndex] = 1
				featureNeuronsWordOrder[sequenceConceptIndex, sequenceFeatureIndex] = j
				featureNeuronsPos[sequenceConceptIndex, sequenceFeatureIndex] = featurePos
	
	featureNeuronsSegmentMask = featureNeuronsSegmentMask.swapdims(0, 1)	#swap from dims [c, s] to [s, c] (in line with featureNeuronsActive)
	#print("featureNeuronsSegmentMask = ", featureNeuronsSegmentMask)	
	
	if(train):
		processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex)
	else:
		firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex = identifySeedIndices(sequenceObservedColumns, sequenceIndex, startIndices, endIndices, sequence, words, lemmas, posTags, conceptIndices, firstSeedTokenIndex, numSeedTokens)
		processFeaturesActiveSeed(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, firstSeedTokenIndex, numSeedTokens, firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex)

def identifySeedIndices(sequenceObservedColumns, sequenceIndex, startIndices, endIndices, sequence, words, lemmas, posTags, conceptIndices, firstSeedTokenIndex, numSeedTokens):
	firstSeedConceptIndex = None
	numSeedConcepts = None
	foundFirstSeedConcept = False
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureWord = words[firstSeedTokenIndex]
		if(useDedicatedConceptNames and firstSeedTokenIndex in sequenceObservedColumns.observedColumnsSequenceWordIndexDict):	
			firstSeedFeatureIndex = featureIndexConceptNeuron
		elif(featureWord in sequenceObservedColumns.featureWordToIndex):
			firstSeedFeatureWord = words[firstSeedTokenIndex]
			firstSeedFeatureIndex = sequenceObservedColumns.databaseNetworkObject.conceptFeaturesDict[firstSeedFeatureWord]
	else:
		firstSeedFeatureIndex = None

	conceptIndicesList = conceptIndices.tolist()
	for i, sequenceConceptWordIndex in enumerate(conceptIndicesList):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			sequenceConceptIndex = i
		else:
			conceptLemma = lemmas[sequenceConceptWordIndex]
			sequenceConceptIndex = sequenceObservedColumns.conceptNameToIndex[conceptLemma] 

		lastWordIndexSeedPhase = firstSeedTokenIndex+numSeedTokens-1
		if(not foundFirstSeedConcept):
			if(firstSeedTokenIndex >= startIndices[sequenceConceptIndex] and firstSeedTokenIndex < endIndices[sequenceConceptIndex]):
				foundFirstSeedConcept = True
				firstSeedConceptIndex = sequenceConceptIndex
				if(inferenceSeedTargetActivationsGlobalFeatureArrays):
					observedColumn = sequenceObservedColumns.observedColumnsSequenceWordIndexDict[sequenceConceptWordIndex]
					sequenceObservedColumns.featureConnections = observedColumn.featureConnections
		if(foundFirstSeedConcept):
			if(lastWordIndexSeedPhase >= startIndices[sequenceConceptIndex] and lastWordIndexSeedPhase < endIndices[sequenceConceptIndex]):
				lastSeedConceptIndex = sequenceConceptIndex
				numSeedConcepts = lastSeedConceptIndex-firstSeedConceptIndex+1
					
	return firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex
	
#first dim cs1 pertains to every concept node in sequence
def processFeaturesActiveSeed(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, firstSeedTokenIndex, numSeedTokens, firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex):
	featureNeuronsInactive = 1 - featureNeuronsActive
	
	fs2 = fs
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		cs2 = sequenceObservedColumns.databaseNetworkObject.c
		featureConnectionsActive = pt.ones(cs, fs, cs2, fs2)
	else:
		cs2 = cs
		featureConnectionsActive, featureConnectionsSegmentMask = createFeatureConnectionsActiveTrain(featureNeuronsActive[arrayIndexSegmentInternalColumn], cs, fs, columnsWordOrder, featureNeuronsWordOrder)

	firstWordIndexPredictPhase = firstSeedTokenIndex+numSeedTokens
	firstConceptIndexPredictPhase = firstSeedConceptIndex+numSeedConcepts
	featureConnectionsActive = createFeatureConnectionsActiveSeed(featureConnectionsActive, cs, fs, cs2, fs2, columnsWordOrder, featureNeuronsWordOrder, firstSeedTokenIndex, firstWordIndexPredictPhase, firstSeedConceptIndex, firstConceptIndexPredictPhase)

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureConnectionsActivationUpdate = featureConnectionsActive[:, firstSeedConceptIndex] * sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrength]
	else:
		featureConnectionsActivationUpdate = featureConnectionsActive * sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrength]
	
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureNeuronsTargetActivation = pt.sum(featureConnectionsActivationUpdate, dim=(1))
	else:
		featureNeuronsTargetActivation = pt.sum(featureConnectionsActivationUpdate, dim=(1, 2))
	if(inferenceActivationFunction):
		featureNeuronsTargetActivation = activationFunction(featureNeuronsTargetActivation)
	else:
		featureNeuronsTargetActivation = featureNeuronsTargetActivation*j1
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		globalFeatureNeuronsActivation = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivation]
		globalFeatureNeuronsActivation = globalFeatureNeuronsActivation + featureNeuronsTargetActivation
	else:
		sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation, :, :, :] += featureNeuronsTargetActivation
	
	if(inferenceDecrementActivations):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivation = decrementActivation(globalFeatureNeuronsActivation, activationDecrementSeed)
		else:
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation] = decrementActivationDense(sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation], activationDecrementSeed)
					
	if(inferenceDeactivateNeuronsUponPrediction):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			if(useSANI):
				printe("processFeaturesActiveSeed error: inferenceDeactivateNeuronsUponPrediction:inferenceSeedTargetActivationsGlobalFeatureArrays:useSANI is not yet implemented")
			else:
				indicesToUpdate = pt.tensor([0, firstSeedConceptIndex, firstSeedFeatureIndex]).unsqueeze(0)
				globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.coalesce()
				globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivation, indicesToUpdate, 0)
		else:
			wordOrderMask = pt.logical_and(featureNeuronsWordOrder >= firstSeedTokenIndex, featureNeuronsWordOrder < firstWordIndexPredictPhase)
			columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1).expand(cs, fs)
			columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded1 >= firstSeedConceptIndex, columnsWordOrderExpanded1 < firstConceptIndexPredictPhase)

			wordOrderMask = pt.logical_and(wordOrderMask, columnsWordOrderMask)
			wordOrderMask = wordOrderMask.unsqueeze(0).expand(arrayNumberOfSegments, cs, fs)
			featureNeuronsActiveSource = pt.logical_and(wordOrderMask, featureNeuronsActive > 0)
			featureNeuronsInactiveSource = pt.logical_not(featureNeuronsActiveSource).float()
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation, :, :, :] *= featureNeuronsInactiveSource

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivation)

def createFeatureConnectionsActiveSeed(featureConnectionsActive, cs, fs, cs2, fs2, columnsWordOrder, featureNeuronsWordOrder, firstSeedTokenIndex, firstWordIndexPredictPhase, firstSeedConceptIndex, firstConceptIndexPredictPhase):
	
	if(featureNeuronsWordOrder is not None):	
		featureNeuronsWordOrderExpanded1 = featureNeuronsWordOrder.view(cs, fs, 1, 1).expand(cs, fs, cs2, fs2)
		wordOrderMask = pt.logical_and(featureNeuronsWordOrderExpanded1 >= firstSeedTokenIndex, featureNeuronsWordOrderExpanded1 < firstWordIndexPredictPhase)
		featureConnectionsActive = featureConnectionsActive * wordOrderMask.unsqueeze(0)
	if(columnsWordOrder is not None):
		columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1, 1, 1).expand(cs, fs, cs2, fs2)
		columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded1 >= firstSeedConceptIndex, columnsWordOrderExpanded1 < firstConceptIndexPredictPhase)
		featureConnectionsActive = featureConnectionsActive * columnsWordOrderMask.unsqueeze(0)
	
	return featureConnectionsActive
	
	
#first dim cs1 pertains to every concept node in sequence
def processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex):
	featureNeuronsInactive = 1 - featureNeuronsActive
		
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesStrength, :, :, :] += featureNeuronsActive
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence, :, :, :] += featureNeuronsActive*z1
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation, :, :, :] = 0
	if(inferenceUseNeuronFeaturePropertiesTime):
		sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTime, :, :, :] = 0
	else:
		sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTime, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTime] + featureNeuronsActive*sequenceIndex
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPos, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPos] + featureNeuronsActive*featureNeuronsPos

	featureConnectionsActive, featureConnectionsSegmentMask = createFeatureConnectionsActiveTrain(featureNeuronsActive[arrayIndexSegmentInternalColumn], cs, fs, columnsWordOrder, featureNeuronsWordOrder)
	
	featureConnectionsPos = featureNeuronsPos.view(1, cs, fs, 1, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)

	featureConnectionsInactive = 1 - featureConnectionsActive

	if(trainNormaliseConnectionStrengthWrtContextLength):
		featureNeuronsWordOrder1d = featureNeuronsWordOrder.flatten()
		featureConnectionsDistances = pt.abs(featureNeuronsWordOrder1d.unsqueeze(1) - featureNeuronsWordOrder1d).reshape(cs, fs, cs, fs)
		featureConnectionsProximity = 1/(featureConnectionsDistances + 1) * 10
		featureConnectionsProximity.unsqueeze(0)
		featureConnectionsStrengthUpdate = featureConnectionsActive*featureConnectionsProximity
	else:
		featureConnectionsStrengthUpdate = featureConnectionsActive

	if(trainIncreaseColumnInternalConnectionsStrength):
		csIndices1 = pt.arange(cs).view(1, cs, 1, 1, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)
		csIndices2 = pt.arange(cs).view(1, 1, 1, cs, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)
		columnInternalConnectionsMask = (csIndices1 == csIndices2)
		columnInternalConnectionsMaskOff = pt.logical_not(columnInternalConnectionsMask)
		featureConnectionsStrengthUpdate = columnInternalConnectionsMask.float()*featureConnectionsStrengthUpdate*trainIncreaseColumnInternalConnectionsStrengthModifier + columnInternalConnectionsMaskOff.float()*featureConnectionsStrengthUpdate

	sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrength, :, :, :, :, :] += featureConnectionsStrengthUpdate
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence, :, :, :, :, :] += featureConnectionsActive*z1
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesActivation, :, :, :, :, :] = 0
	if(inferenceUseNeuronFeaturePropertiesTime):
		sequenceObservedColumns.featureConnections[arrayIndexPropertiesTime, :, :, :, :, :] = 0
	else:
		sequenceObservedColumns.featureConnections[arrayIndexPropertiesTime, :, :, :, :, :] = featureConnectionsInactive*sequenceObservedColumns.featureConnections[arrayIndexPropertiesTime] + featureConnectionsActive*sequenceIndex
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPos, :, :, :, :, :] = featureConnectionsInactive*sequenceObservedColumns.featureConnections[arrayIndexPropertiesPos] + featureConnectionsActive*featureConnectionsPos

	if(trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections):
		decreasePermanenceActive(sequenceObservedColumns, featureNeuronsActive[arrayIndexSegmentInternalColumn], featureNeuronsInactive[arrayIndexSegmentInternalColumn], sequenceConceptIndexMask, featureNeuronsSegmentMask, featureConnectionsSegmentMask)
	

def createFeatureConnectionsActiveTrain(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder):

	featureNeuronsActive1d = featureNeuronsActive.view(cs*fs)
	featureConnectionsActive = pt.matmul(featureNeuronsActive1d.unsqueeze(1), featureNeuronsActive1d.unsqueeze(0)).view(cs, fs, cs, fs)

	if(featureNeuronsWordOrder is not None):
		featureNeuronsWordOrderExpanded1 = featureNeuronsWordOrder.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)
		featureNeuronsWordOrderExpanded2 = featureNeuronsWordOrder.view(1, 1, cs, fs).expand(cs, fs, cs, fs)
		wordOrderMask = featureNeuronsWordOrderExpanded2 > featureNeuronsWordOrderExpanded1
		featureConnectionsActive = featureConnectionsActive * wordOrderMask
	if(columnsWordOrder is not None):
		columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
		columnsWordOrderExpanded2 = columnsWordOrder.view(1, 1, cs, 1).expand(cs, fs, cs, fs)
		if(debugConnectColumnsToNextColumnsInSequenceOnly):
			columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded2 >= columnsWordOrderExpanded1, columnsWordOrderExpanded2 <= columnsWordOrderExpanded1+1)
		else:
			columnsWordOrderMask = columnsWordOrderExpanded2 >= columnsWordOrderExpanded1
		featureConnectionsActive = featureConnectionsActive * columnsWordOrderMask
	
	csIndices1 = pt.arange(cs).view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
	csIndices2 = pt.arange(cs).view(1, 1, cs, 1).expand(cs, fs, cs, fs)
	fsIndices1 = pt.arange(fs).view(1, fs, 1, 1).expand(cs, fs, cs, fs)
	fsIndices2 = pt.arange(fs).view(1, 1, 1, fs).expand(cs, fs, cs, fs)
	identityMask = (csIndices1 != csIndices2) | (fsIndices1 != fsIndices2)
	featureConnectionsActive = featureConnectionsActive * identityMask

	if(useSANI):
		featureConnectionsActive, featureConnectionsSegmentMask = assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs)
	else:
		featureConnectionsActive = featureConnectionsActive.unsqueeze(0)
		featureConnectionsSegmentMask = pt.ones_like(featureConnectionsActive)
	
	return featureConnectionsActive, featureConnectionsSegmentMask

def assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs):

	conceptNeuronsConceptOrder1d = pt.arange(cs)
	conceptNeuronsDistances = pt.abs(conceptNeuronsConceptOrder1d.unsqueeze(1) - conceptNeuronsConceptOrder1d).reshape(cs, cs)
	connectionsSegmentIndex = arrayNumberOfSegments-conceptNeuronsDistances-1
	connectionsSegmentIndex = pt.clamp(connectionsSegmentIndex, min=0)
	
	featureConnectionsSegmentMask = pt.zeros((arrayNumberOfSegments, cs, cs), dtype=pt.bool)
	featureConnectionsSegmentMask = featureConnectionsSegmentMask.scatter_(0, connectionsSegmentIndex.unsqueeze(0), True)
	featureConnectionsSegmentMask = featureConnectionsSegmentMask.view(arrayNumberOfSegments, cs, 1, cs, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)
	
	featureConnectionsActive = featureConnectionsSegmentMask * featureConnectionsActive.unsqueeze(0)
	
	return featureConnectionsActive, featureConnectionsSegmentMask
		
def decreasePermanenceActive(sequenceObservedColumns, featureNeuronsActive, featureNeuronsInactive, sequenceConceptIndexMask, featureNeuronsSegmentMask, featureConnectionsSegmentMask):

	if(trainSequenceObservedColumnsMatchSequenceWords):
		featureNeuronsInactive = featureNeuronsInactive*sequenceConceptIndexMask
	
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs 
	
	featureNeuronsDecrease = featureNeuronsInactive.unsqueeze(0)*z2 * featureNeuronsSegmentMask.unsqueeze(2)
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence, :, :, :] -= featureNeuronsDecrease
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence] = pt.clamp(sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence], min=0)

	featureNeuronsAll = pt.ones((cs, fs), dtype=arrayType)
	featureNeuronsAll1d = featureNeuronsAll.view(cs*fs)
	featureNeuronsActive1d = featureNeuronsActive.view(cs*fs)
	featureNeuronsInactive1d = featureNeuronsInactive.view(cs*fs)
	 
	featureConnectionsDecrease1 = pt.matmul(featureNeuronsInactive1d.unsqueeze(1), featureNeuronsAll1d.unsqueeze(0)).view(cs, fs, cs, fs)
	featureConnectionsDecrease1 = featureConnectionsDecrease1.unsqueeze(0)*featureConnectionsSegmentMask
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence, :, :, :, :, :] -= featureConnectionsDecrease1
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence] = pt.clamp(sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence], min=0)
	
	featureConnectionsDecrease2 = pt.matmul(featureNeuronsActive1d.unsqueeze(1), featureNeuronsInactive1d.unsqueeze(0)).view(cs, fs, cs, fs)
	featureConnectionsDecrease2 = featureConnectionsDecrease2.unsqueeze(0)*featureConnectionsSegmentMask
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence, :, :, :, :, :] -= featureConnectionsDecrease2
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence] = pt.clamp(sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence], min=0)
 

def decrementActivationDense(featureNeuronsActivation, activationDecrement):
	if(inferenceDecrementActivationsNonlinear):
		featureNeuronsActivation = featureNeuronsActivation * (1-activationDecrement)
	else:
		featureNeuronsActivation = featureNeuronsActivation - activationDecrementPerPredictedSequence
	return featureNeuronsActivation


def decrementActivation(featureNeuronsActivation, activationDecrement):
	if(inferenceDecrementActivationsNonlinear):
		featureNeuronsActivation = featureNeuronsActivation * (1-activationDecrement)
	else:
		featureNeuronsActivation = GIAANNproto_sparseTensors.subtractValueFromSparseTensorValues(featureNeuronsActivation, activationDecrementPerPredictedSequence)
	return featureNeuronsActivation


def activationFunction(x):
	'''
	A non-linear activation function similar to a sigmoid that outputs from 0 to +1, but the slope of the function goes to 0 at approx 50 instead of 5. 
	The function outputs 0 when the input is 0. All input will be positive. 
	'''
	if x.is_sparse:
		indices = x._indices()
		values = x._values()
		transformedValues = hybridActivation(values)
		z = pt.sparse_coo_tensor(indices, transformedValues, x.size(), device=x.device)
	else:
		z = hybridActivation(x)
	return z

def hybridActivation(x, scale=100.0):
	#print("x = ", x)
	f = (pt.sigmoid(x / scale) - 0.5 ) * 2.0
	#print("f = ", f)
	return f

def getLemmas(sequence):
	words = []
	lemmas = []
	posTags = []
	
	for token in sequence:
		word = token.text.lower()
		lemma = token.lemma_.lower()
		pos = token.pos_  # Part-of-speech tag
		words.append(word)
		lemmas.append(lemma)
		posTags.append(pos)
	
	return words, lemmas, posTags





#low level processFeaturesActivePredict functions currently stored here (shared):

#first dim cs1 restricted to a candiate set of tokens.
def processFeaturesActivePredictMulti(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices):
	#print("processFeaturesActivePredictMulti:")
	for conceptIndex in range(conceptColumnsIndices.shape[0]):
		conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex].unsqueeze(dim=0)
		conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndices[conceptIndex].unsqueeze(dim=0)
		featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 2, conceptIndex)	#sequence concept index dimension	#CHECKTHIS
		globalFeatureNeuronsActivation, globalFeatureConnectionsActivation = processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource)
	
	return globalFeatureNeuronsActivation, globalFeatureConnectionsActivation
	
#first dim cs1 restricted to a single token
def processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices):
	featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 2, 0)	#sequence concept index dimension
	return processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices)

def processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices):
		
	featureNeuronsActive = GIAANNproto_sparseTensors.neuronActivationSparse(globalFeatureNeuronsActivation, algorithmMatrixSANImethod)
	
	featureNeuronsActive = featureNeuronsActive[conceptColumnsIndices.squeeze().item()]	#select columns
	featureNeuronsActive = featureNeuronsActive[conceptColumnsFeatureIndices.squeeze().squeeze().item()]	#select features
	
	#target neuron activation dependence on connection strength;
	featureConnections = featureConnections[arrayIndexPropertiesStrength]
	if(inferencePredictiveNetwork and not useGPUsparse):
		conceptColumnsFeatureIndices = conceptColumnsFeatureIndices.to(deviceSparse)
	featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnections, 1, conceptColumnsFeatureIndices.squeeze().item())
	if(inferenceConnectionsStrengthBoolean):
		featureConnections = featureConnections.bool().float()
	
	featureNeuronsTargetActivation = featureNeuronsActive * featureConnections

	if(inferenceActivationFunction):
		featureNeuronsTargetActivation = activationFunction(featureNeuronsTargetActivation)
		#print("featureNeuronsTargetActivation = ", featureNeuronsTargetActivation)
	else:
		featureNeuronsTargetActivation = featureNeuronsTargetActivation*j1
		
	#update the activations of the target nodes;
	if(not useSANI or algorithmMatrixSANImethod=="doNotEnforceSequentialityAcrossSegments"):
		globalFeatureNeuronsActivation += featureNeuronsTargetActivation
	elif(algorithmMatrixSANImethod=="enforceSequentialActivationAcrossSegments"):
		globalFeatureNeuronsActivationDense = globalFeatureNeuronsActivation.to_dense()
		featureNeuronsTargetActivationDense = featureNeuronsTargetActivation.to_dense()
		previousChannelActivation = globalFeatureNeuronsActivationDense[:-1] > 0	
		globalFeatureNeuronsActivationDense[1:] += featureNeuronsTargetActivationDense[1:] * previousChannelActivation
		globalFeatureNeuronsActivationDense[0] += featureNeuronsTargetActivationDense[0]
		globalFeatureNeuronsActivation = globalFeatureNeuronsActivationDense.to_sparse_coo()
	if(inferenceActivationStrengthBoolean):
		globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.bool().float()
		
	if(transformerUseInputConnections):
		featureNeuronsTargetActivation = GIAANNproto_sparseTensors.expand_sparse_tensor(featureNeuronsTargetActivation, 1, conceptColumnsIndices.squeeze(), new_dim_size=databaseNetworkObject.c)
		featureNeuronsTargetActivation = GIAANNproto_sparseTensors.expand_sparse_tensor(featureNeuronsTargetActivation, 2, conceptColumnsFeatureIndices.squeeze(), new_dim_size=databaseNetworkObject.f)
		globalFeatureConnectionsActivation = globalFeatureConnectionsActivation + featureNeuronsTargetActivation

	return globalFeatureNeuronsActivation, globalFeatureConnectionsActivation
