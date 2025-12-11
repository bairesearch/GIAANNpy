"""GIAANNproto_predictionActivate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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

#first dim cs1 restricted to a candiate set of tokens.
def processFeaturesActivePredictMulti(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices):
	#print("processFeaturesActivePredictMulti:")
	for conceptIndex in range(conceptColumnsIndices.shape[0]):
		conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex].unsqueeze(dim=0)
		conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndices[conceptIndex].unsqueeze(dim=0)
		sourceConceptIndexValue = conceptColumnsIndicesSource.squeeze().item()
		featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 2, conceptIndex)	#sequence concept index dimension	#CHECKTHIS
		globalFeatureNeuronsActivation, globalFeatureConnectionsActivation = processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource, sourceConceptIndexValue)
	
	return globalFeatureNeuronsActivation, globalFeatureConnectionsActivation
	
#first dim cs1 restricted to a single token
def processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices):
	featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 2, 0)	#sequence concept index dimension
	sourceConceptIndexValue = conceptColumnsIndices.squeeze().item()
	return processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices, sourceConceptIndexValue)

def processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices, sourceConceptIndex=None):
		
	featureNeuronsActive = GIAANNproto_sparseTensors.neuronActivationSparse(globalFeatureNeuronsActivation, algorithmMatrixSANImethod)
	
	featureNeuronsActive = featureNeuronsActive[conceptColumnsIndices.squeeze().item()]	#select columns
	featureNeuronsActive = featureNeuronsActive[conceptColumnsFeatureIndices.squeeze().squeeze().item()]	#select features
	
	#target neuron activation dependence on connection strength;
	featureConnectionsStrength = featureConnections[arrayIndexPropertiesStrength]
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = featureConnections[arrayIndexPropertiesPos]
	if(inferencePredictiveNetwork and not useGPUsparse):
		conceptColumnsFeatureIndices = conceptColumnsFeatureIndices.to(deviceSparse)
	featureConnectionsStrength = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsStrength, 1, conceptColumnsFeatureIndices.squeeze().item())
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsPos, 1, conceptColumnsFeatureIndices.squeeze().item())
		featureConnectionsStrength = applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceConceptIndex)
	if(inferenceConnectionsStrengthBoolean):
		featureConnectionsStrength = featureConnectionsStrength.bool().float()
	
	featureNeuronsTargetActivation = featureNeuronsActive * featureConnectionsStrength

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
			scopeMask = (strengthIndices[1] != sourceConceptIndex)
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
	if(enforceDirectConnections and enforceDirectConnectionsMinWordDistance):
		if(targetIndices is None or targetIndices.shape[1] == 0):
			printe("(targetIndices is None or targetIndices.shape[1] == 0)")
			#return None
		featureConnectionsMinWordDistance = observedColumn.featureConnections[arrayIndexPropertiesMinWordDistanceIndex]
		featureConnectionsMinWordDistance = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsMinWordDistance, 1, sourceFeatureIndex)
		featureConnectionsMinWordDistance = featureConnectionsMinWordDistance.coalesce()
		if(featureConnectionsMinWordDistance._nnz() == 0):
			printe("(featureConnectionsMinWordDistance._nnz() == 0)")
			#return pt.zeros(targetIndices.shape[1], dtype=pt.bool, device=targetIndices.device)
		minIndices = featureConnectionsMinWordDistance.indices()
		minValues = featureConnectionsMinWordDistance.values()
		minDistanceLookup = {}
		for idx in range(minValues.shape[0]):
			columnValue = int(minIndices[1, idx].item())
			featureValue = int(minIndices[2, idx].item())
			distanceValue = float(minValues[idx].item())
			key = (columnValue, featureValue)
			if(key not in minDistanceLookup or distanceValue < minDistanceLookup[key]):
				minDistanceLookup[key] = distanceValue
		maskList = []
		for idx in range(targetIndices.shape[1]):
			columnValue = int(targetIndices[1, idx].item())
			featureValue = int(targetIndices[2, idx].item())
			distanceValue = minDistanceLookup.get((columnValue, featureValue))
			if(distanceValue is None):
				maskList.append(False)
			else:
				maskList.append(abs(distanceValue - requiredDistance) < 1e-4)
		if(len(maskList) == 0):
			mask = pt.zeros(0, dtype=pt.bool, device=targetIndices.device)
		else:
			mask = pt.tensor(maskList, dtype=pt.bool, device=targetIndices.device)
		#print("mask = ", mask)
	else:
		mask = None
	return mask
	
