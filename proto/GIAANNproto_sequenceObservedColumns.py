"""GIAANNproto_sequenceObservedColumns.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors

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

		# Initialize arrays
		self.featureNeurons = self.initialiseFeatureNeuronsSequence(self.cs, self.fs)
		self.featureConnections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

		# Populate arrays with data from observedColumnsDict
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


