"""GIAANNproto_predictionActivate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto Prediction Activate

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors


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

if(inferenceSegmentActivationsBoolean):
	def applySegmentActivationsBooleanFeatureSegmentsOnly(globalFeatureNeuronsActivation):
		featureSegmentStart = arrayNumberOfSegmentsColumnDistance
		if featureSegmentStart <= 0:
			printe("useSANIfeaturesAndColumns and no feature segments")
		if globalFeatureNeuronsActivation.is_sparse:
			sparseActivation = globalFeatureNeuronsActivation.coalesce()
			if featureSegmentStart >= sparseActivation.size(1):
				printe("feature segments start beyond the last segment")
			indices = sparseActivation.indices()
			values = sparseActivation.values()
			mask = indices[1] >= featureSegmentStart
			if pt.any(mask):
				#entries in feature segments
				values = values.clone()
				values[mask] = (values[mask] > 0).to(values.dtype)
				return pt.sparse_coo_tensor(indices, values, sparseActivation.size(), device=sparseActivation.device)
			else:
				return sparseActivation
		else:
			globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.clone()
			if featureSegmentStart < globalFeatureNeuronsActivation.size(1):
				globalFeatureNeuronsActivation[:, featureSegmentStart:] = (globalFeatureNeuronsActivation[:, featureSegmentStart:] > 0).to(globalFeatureNeuronsActivation.dtype)
			return globalFeatureNeuronsActivation

	def applySegmentActivationsBoolean(globalFeatureNeuronsActivation):
		if(inferenceSegmentActivationsBooleanFeatureSegmentsOnly):
			if(useSANIcolumns):
				return globalFeatureNeuronsActivation
			elif(useSANIfeatures):
				return globalFeatureNeuronsActivation.bool().float()
			elif(useSANIfeaturesAndColumns):
				return applySegmentActivationsBooleanFeatureSegmentsOnly(globalFeatureNeuronsActivation)
		else:
			return globalFeatureNeuronsActivation.bool().float()

#first dim cs1 restricted to a candiate set of tokens.
def processFeaturesActivePredictMulti(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices):
	#print("processFeaturesActivePredictMulti:")
	for conceptIndex in range(conceptColumnsIndices.shape[0]):
		conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex].unsqueeze(dim=0)
		conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndices[conceptIndex].unsqueeze(dim=0)
		sourceConceptIndexValue = conceptColumnsIndicesSource.squeeze().item()
		featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 3, conceptIndex)	#sequence concept index dimension	#CHECKTHIS
		globalFeatureNeuronsActivation, globalFeatureConnectionsActivation = processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource, sourceConceptIndexValue)
	
	return globalFeatureNeuronsActivation, globalFeatureConnectionsActivation
	
#first dim cs1 restricted to a single token
def processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices):
	featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 3, 0)	#sequence concept index dimension
	sourceConceptIndexValue = conceptColumnsIndices.squeeze().item()
	return processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices, sourceConceptIndexValue)

def processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices, sourceConceptIndex=None):
		
	featureNeuronsActive = GIAANNproto_sparseTensors.neuronActivationSparse(globalFeatureNeuronsActivation, algorithmMatrixSANImethod)
	
	sourceColumnIndex = conceptColumnsIndices.squeeze().item()
	sourceFeatureIndex = conceptColumnsFeatureIndices.squeeze().squeeze().item()
	if(featureNeuronsActive.is_sparse):
		featureNeuronsActive = featureNeuronsActive.coalesce()
		if(featureNeuronsActive.dim() == 3):
			branchCount = featureNeuronsActive.size(0)
			indices = featureNeuronsActive.indices()
			values = featureNeuronsActive.values()
			mask = (indices[1] == sourceColumnIndex) & (indices[2] == sourceFeatureIndex)
			featureNeuronsActiveDense = pt.zeros((branchCount,), dtype=values.dtype, device=values.device)
			if(mask.any()):
				branchIndices = indices[0, mask]
				featureNeuronsActiveDense.index_add_(0, branchIndices, values[mask])
			featureNeuronsActive = featureNeuronsActiveDense
		elif(featureNeuronsActive.dim() == 2):
			indices = featureNeuronsActive.indices()
			values = featureNeuronsActive.values()
			mask = (indices[0] == sourceColumnIndex) & (indices[1] == sourceFeatureIndex)
			if(mask.any()):
				featureNeuronsActive = values[mask].sum()
			else:
				featureNeuronsActive = pt.zeros((), dtype=values.dtype, device=values.device)
		else:
			featureNeuronsActive = featureNeuronsActive.to_dense()
			if(featureNeuronsActive.dim() == 3):
				featureNeuronsActive = featureNeuronsActive[:, sourceColumnIndex, sourceFeatureIndex]
			else:
				featureNeuronsActive = featureNeuronsActive[sourceColumnIndex, sourceFeatureIndex]
	else:
		if(featureNeuronsActive.dim() == 3):
			featureNeuronsActive = featureNeuronsActive[:, sourceColumnIndex, sourceFeatureIndex]
		else:
			featureNeuronsActive = featureNeuronsActive[sourceColumnIndex, sourceFeatureIndex]
	if(inferenceSourceActivationsBoolean):
		featureNeuronsActive = (featureNeuronsActive > 0).to(featureNeuronsActive.dtype)	#ensure the source activation signal is binary (even with useSANI)
	if(multipleDendriticBranches and featureNeuronsActive.dim() == 1):
		# Collapse branch-local source activations so each target branch receives the same drive.
		featureNeuronsActive = featureNeuronsActive.sum()

	#target neuron activation dependence on connection strength;
	featureConnectionsStrength = featureConnections[arrayIndexPropertiesStrengthIndex]
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = featureConnections[arrayIndexPropertiesPosIndex]
	if(inferencePredictiveNetwork and not useGPUsparse):
		conceptColumnsFeatureIndices = conceptColumnsFeatureIndices.to(deviceSparse)
	featureConnectionsStrength = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsStrength, 2, sourceFeatureIndex)
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsPos, 2, sourceFeatureIndex)
		featureConnectionsStrength = applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceConceptIndex)
	if(inferenceConnectionsStrengthBoolean):
		featureConnectionsStrength = featureConnectionsStrength.bool().float()
	
	if(featureNeuronsActive.dim() > 0):
		featureNeuronsActive = featureNeuronsActive.reshape(-1)
	if featureConnectionsStrength.is_sparse:
		if(featureNeuronsActive.dim() == 0):
			branchCount = featureConnectionsStrength.size(0)
			branchValues = pt.full((branchCount,), featureNeuronsActive.item(), dtype=featureNeuronsActive.dtype, device=featureNeuronsActive.device)
			featureNeuronsTargetActivation = GIAANNproto_sparseTensors.scaleSparseTensorByBranchValues(featureConnectionsStrength, branchValues)
		else:
			featureNeuronsTargetActivation = GIAANNproto_sparseTensors.scaleSparseTensorByBranchValues(featureConnectionsStrength, featureNeuronsActive)
	else:
		if(featureNeuronsActive.dim() == 0):
			featureNeuronsTargetActivation = featureConnectionsStrength * featureNeuronsActive
		else:
			featureNeuronsTargetActivation = featureConnectionsStrength * featureNeuronsActive.view(-1, 1, 1, 1)

	if(inferenceActivationFunction):
		featureNeuronsTargetActivation = activationFunction(featureNeuronsTargetActivation)
		#print("featureNeuronsTargetActivation = ", featureNeuronsTargetActivation)
	else:
		featureNeuronsTargetActivation = featureNeuronsTargetActivation*j1

	#update the activations of the target nodes;
	if(useSANI):
		if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
			if(enforceSequentialActivation):
				globalFeatureNeuronsActivationDense = globalFeatureNeuronsActivation.to_dense()
				featureNeuronsTargetActivationDense = featureNeuronsTargetActivation
				if(featureNeuronsTargetActivationDense.is_sparse):
					featureNeuronsTargetActivationDense = featureNeuronsTargetActivationDense.to_dense()
				for branchIndex in range(globalFeatureNeuronsActivationDense.shape[0]):
					branchActivation = globalFeatureNeuronsActivationDense[branchIndex]
					branchTargetActivation = featureNeuronsTargetActivationDense[branchIndex]
					if(useSANIfeaturesAndColumns):
						# For useSANIfeaturesAndColumns, enforce sequential activation independently for:
						# a) concept/column segments and b) feature segments.
						featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
						assert featureSegmentsOffset >= 0 and featureSegmentsOffset < arrayNumberOfSegments
						previousConceptChannelActivation = branchActivation[:featureSegmentsOffset-1] > 0 if featureSegmentsOffset > 1 else None
						previousFeatureChannelActivation = branchActivation[featureSegmentsOffset:arrayNumberOfSegments-1] > 0 if featureSegmentsOffset+1 < arrayNumberOfSegments else None
						if(previousConceptChannelActivation is not None):
							branchActivation[1:featureSegmentsOffset] += branchTargetActivation[1:featureSegmentsOffset] * previousConceptChannelActivation
						if(previousFeatureChannelActivation is not None):
							branchActivation[featureSegmentsOffset+1:] += branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
						branchActivation[0] += branchTargetActivation[0]
						branchActivation[featureSegmentsOffset] += branchTargetActivation[featureSegmentsOffset]
					else:
						previousChannelActivation = branchActivation[:-1] > 0
						branchActivation[1:] += branchTargetActivation[1:] * previousChannelActivation
						branchActivation[0] += branchTargetActivation[0]
					globalFeatureNeuronsActivationDense[branchIndex] = branchActivation
				globalFeatureNeuronsActivation = globalFeatureNeuronsActivationDense.to_sparse_coo()
			else:
				globalFeatureNeuronsActivation += featureNeuronsTargetActivation
		elif(algorithmMatrixSANImethod=="doNotEnforceActivationAcrossSegments"):
			globalFeatureNeuronsActivation += featureNeuronsTargetActivation
	else:
		globalFeatureNeuronsActivation += featureNeuronsTargetActivation
	if(inferenceSegmentActivationsBoolean):
		globalFeatureNeuronsActivation = applySegmentActivationsBoolean(globalFeatureNeuronsActivation)
		
	if(transformerUseInputConnections):
		featureNeuronsTargetActivation = GIAANNproto_sparseTensors.expand_sparse_tensor(featureNeuronsTargetActivation, 2, conceptColumnsIndices.squeeze(), new_dim_size=databaseNetworkObject.c)
		featureNeuronsTargetActivation = GIAANNproto_sparseTensors.expand_sparse_tensor(featureNeuronsTargetActivation, 3, conceptColumnsFeatureIndices.squeeze(), new_dim_size=databaseNetworkObject.f)
		globalFeatureConnectionsActivation = globalFeatureConnectionsActivation + featureNeuronsTargetActivation

	return globalFeatureNeuronsActivation, globalFeatureConnectionsActivation

def selectActivatedBranchIndex(globalFeatureNeuronsActivation, columnIndex, featureIndex):
	if(not multipleDendriticBranches):
		return 0
	if(globalFeatureNeuronsActivation is None):
		return 0
	if(globalFeatureNeuronsActivation.is_sparse):
		sparseActivation = globalFeatureNeuronsActivation.coalesce()
		if(sparseActivation._nnz() == 0):
			return 0
		indices = sparseActivation.indices()
		values = sparseActivation.values()
		mask = (indices[2] == columnIndex) & (indices[3] == featureIndex)
		if(not pt.any(mask)):
			return 0
		branchIndices = indices[0, mask].tolist()
		branchValues = values[mask].tolist()
		branchScores = {}
		for branchIndex, value in zip(branchIndices, branchValues):
			branchScores[branchIndex] = branchScores.get(branchIndex, 0.0) + float(value)
		bestBranch = max(branchScores, key=branchScores.get)
		return int(bestBranch)
	activationSlice = globalFeatureNeuronsActivation[:, :, columnIndex, featureIndex]
	if(activationSlice.numel() == 0):
		return 0
	branchScores = activationSlice.sum(dim=1)
	bestBranch = int(pt.argmax(branchScores).item())
	return bestBranch

def applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceConceptIndex):
	posLookup = getConnectionStrengthPOSdependenceLookup(databaseNetworkObject)
	if posLookup:
		featureConnectionsStrength = featureConnectionsStrength.coalesce()
		if featureConnectionsStrength._nnz() == 0:
			printe("featureConnectionsStrength._nnz() == 0")
			#return featureConnectionsStrength
		if featureConnectionsPos is None:
			printe("featureConnectionsPos is None")
			#return featureConnectionsStrength
		featureConnectionsPos = featureConnectionsPos.coalesce()
		if featureConnectionsPos._nnz() == 0:
			printe("featureConnectionsPos._nnz() == 0")
			#return featureConnectionsStrength
		strengthIndices = featureConnectionsStrength.indices()
		strengthValues = featureConnectionsStrength.values()
		posIndices = featureConnectionsPos.indices()
		posValues = featureConnectionsPos.values()
		if strengthIndices.shape[1] == posIndices.shape[1] and pt.equal(strengthIndices, posIndices):
			alignedPosValues = posValues
		else:
			posIndicesCPU = posIndices.cpu()
			posValuesCPU = posValues.cpu()
			posIndexMap = {tuple(posIndicesCPU[:, idx].tolist()): posValuesCPU[idx].item() for idx in range(posIndicesCPU.shape[1])}
			strengthIndicesCPU = strengthIndices.cpu()
			alignedPosList = []
			for idx in range(strengthIndicesCPU.shape[1]):
				key = tuple(strengthIndicesCPU[:, idx].tolist())
				alignedPosList.append(posIndexMap.get(key, 0.0))
			alignedPosValues = pt.tensor(alignedPosList, dtype=posValues.dtype, device=posValues.device)
		alignedPosValues = alignedPosValues.long()
		if(connectionStrengthPOSdependenceExternal and sourceConceptIndex is not None):
			scopeMask = (strengthIndices[2] != sourceConceptIndex)
		else:
			scopeMask = pt.ones(strengthIndices.shape[1], dtype=pt.bool, device=strengthValues.device)
		if not pt.any(scopeMask):
			return featureConnectionsStrength
		else:
			scaleTensor = pt.ones_like(strengthValues)
			for posIndex, scaleValue in posLookup:
				if scaleValue == 1:
					continue
				posMask = (alignedPosValues == posIndex) & scopeMask
				if pt.any(posMask):
					scaleTensor[posMask] = scaleValue
			strengthValues *= scaleTensor
	return featureConnectionsStrength

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

def computeConnectionMinWordDistanceMask(observedColumn, sourceFeatureIndex, targetIndices, requiredDistance=1.0):
	if(enforceDirectConnectionsMinWordDistance):
		if(targetIndices is None or targetIndices.shape[1] == 0):
			printe("(targetIndices is None or targetIndices.shape[1] == 0)")
			#return None
		featureConnectionsMinWordDistance = observedColumn.featureConnections[arrayIndexPropertiesMinWordDistanceIndex]
		featureConnectionsMinWordDistance = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsMinWordDistance, 2, sourceFeatureIndex)
		featureConnectionsMinWordDistance = featureConnectionsMinWordDistance.coalesce()
		if(featureConnectionsMinWordDistance._nnz() == 0):
			printe("(featureConnectionsMinWordDistance._nnz() == 0)")
			#return pt.zeros(targetIndices.shape[1], dtype=pt.bool, device=targetIndices.device)
		minIndices = featureConnectionsMinWordDistance.indices()
		minValues = featureConnectionsMinWordDistance.values()
		minDistanceLookup = {}
		for idx in range(minValues.shape[0]):
			columnValue = int(minIndices[2, idx].item())
			featureValue = int(minIndices[3, idx].item())
			distanceValue = float(minValues[idx].item())
			key = (columnValue, featureValue)
			if(key not in minDistanceLookup or distanceValue < minDistanceLookup[key]):
				minDistanceLookup[key] = distanceValue
		maskList = []
		for idx in range(targetIndices.shape[1]):
			columnValue = int(targetIndices[2, idx].item())
			featureValue = int(targetIndices[3, idx].item())
			distanceValue = minDistanceLookup.get((columnValue, featureValue))
			if(distanceValue is None):
				maskList.append(False)
			else:
				maskList.append(abs(distanceValue - requiredDistance) < 1e-4)
		if(len(maskList) == 0):
			mask = pt.zeros(0, dtype=pt.bool, device=targetIndices.device)
		else:
			mask = pt.tensor(maskList, dtype=pt.bool, device=targetIndices.device)
		if(debugPrintMinWordDistanceDetails):
			printMinWordDistanceDetails(observedColumn, sourceFeatureIndex, targetIndices, mask, minDistanceLookup)
		#print("mask = ", mask)
	else:
		mask = None
	return mask
	
def printMinWordDistanceDetails(observedColumn, sourceFeatureIndex, targetIndices, mask, minDistanceLookup):
	databaseNetworkObject = getattr(observedColumn, "databaseNetworkObject", None)
	sourceColumnName = getattr(observedColumn, "conceptName", "<unknown>")
	if(databaseNetworkObject is not None and 0 <= sourceFeatureIndex < len(databaseNetworkObject.conceptFeaturesList)):
		sourceFeatureName = databaseNetworkObject.conceptFeaturesList[sourceFeatureIndex]
	else:
		sourceFeatureName = f"<feature:{sourceFeatureIndex}>"
	for idx in range(targetIndices.shape[1]):
		columnValue = int(targetIndices[2, idx].item())
		featureValue = int(targetIndices[2, idx].item())
		distanceValue = minDistanceLookup.get((columnValue, featureValue))
		keepConnection = (mask[idx].item() == 1) if (mask is not None and idx < mask.shape[0]) else False
		columnName = f"<column:{columnValue}>"
		featureName = f"<feature:{featureValue}>"
		if(databaseNetworkObject is not None):
			if(0 <= columnValue < len(databaseNetworkObject.conceptColumnsList)):
				columnName = databaseNetworkObject.conceptColumnsList[columnValue]
			if(0 <= featureValue < len(databaseNetworkObject.conceptFeaturesList)):
				featureName = databaseNetworkObject.conceptFeaturesList[featureValue]
		print(f"debugMinDistance: source {sourceColumnName}:{sourceFeatureName} -> target {columnName}:{featureName} distance={distanceValue} keep={keepConnection}")
