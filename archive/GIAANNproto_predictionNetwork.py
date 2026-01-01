"""GIAANNproto_predictionNetwork.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto prediction Network

"""

import torch as pt

from GIAANNproto_globalDefs import *
if(inferencePredictiveNetwork):
	if(inferencePredictiveNetworkModel=="ColumnMLP"):
		import GIAANNproto_predictionNetworkModelColumnMLP as GIAANNproto_predictionModel
	elif(inferencePredictiveNetworkModel=="MLP"):
		import GIAANNproto_predictionNetworkModelMLP as GIAANNproto_predictionModel
	elif(inferencePredictiveNetworkModel=="Transformer"):
		import GIAANNproto_predictionNetworkModelTransformer as GIAANNproto_predictionModel
	import GIAANNproto_predictionNetworkOperations

def inferenceSavePredictiveNetwork():
	GIAANNproto_predictionModel.saveModel(predictiveNetworkFolder, predictiveNetworkFileName)

def initialisePredictiveNetwork(databaseNetworkObject):
	GIAANNproto_predictionModel.nextWordPredictionModelCreate(databaseNetworkObject)

def predictMostActiveFeature(sequenceObservedColumns, databaseNetworkObject, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, conceptActivationState=None, connectedColumns=None, connectedColumnsFeatures=None):		
	#generate targets;
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		targets = None
		targetsC = pt.zeros(databaseNetworkObject.c)
		targetsF = pt.zeros(databaseNetworkObject.f)
		targetsC[targetPreviousColumnIndex] = 1
		targetsF[targetFeatureIndex] = 1
		if(targetMultipleSources):
			targetsC[targetNextColumnIndex] = 1	
	else:
		targets = pt.zeros(databaseNetworkObject.c, databaseNetworkObject.f)
		targets[targetPreviousColumnIndex, targetFeatureIndex] = 1
		if(targetMultipleSources):
			targets[targetNextColumnIndex, targetFeatureIndex] = 1
		targetsC = None
		targetsF = None
	
	globalFeatureConnections = None
	if(inferencePredictiveNetworkUseInputAllProperties):
		globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons
		if(transformerUseInputConnections):
			globalFeatureConnections = databaseNetworkObject.globalFeatureConnections
	else:
		globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
		if(transformerUseInputConnections):
			globalFeatureConnections = databaseNetworkObject.globalFeatureConnections[arrayIndexPropertiesActivationIndex]
	
	if(inferencePredictiveNetworkNormaliseInputs):
		#if(not useGPUpredictiveNetworkModel and inferencePredictiveNetworkNormaliseDim==0):
		#	globalFeatureNeurons = GIAANNproto_predictionNetworkOperations.normaliseSparseTensor(globalFeatureNeurons, inferencePredictiveNetworkUseInputAllProperties)
		if(transformerUseInputConnections):	#globalFeatureConnections are currently retained on CPU
			globalFeatureConnections = GIAANNproto_predictionNetworkOperations.normaliseSparseTensor(globalFeatureConnections, inferencePredictiveNetworkUseInputAllProperties)
			if(inferencePredictiveNetworkNormaliseDim != 0):
				print("predictMostActiveFeature warning: inferencePredictiveNetworkNormaliseDim>0 - can only normalise globalFeatureConnections along first dimension (properties)")
				
	if(inferencePredictiveNetworkModel=="ColumnMLP"):
		#GIAANNproto_predictionModel.ensureModelMatchesDatabase(databaseNetworkObject)
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictionModel.nextWordPredictionColumnMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF)
	elif(inferencePredictiveNetworkModel=="MLP"):
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictionModel.nextWordPredictionMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF)
	elif(inferencePredictiveNetworkModel=="Transformer"):
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictionModel.nextWordPredictionTransformerTrainStep(globalFeatureNeurons, globalFeatureConnections, targets, targetsC, targetsF)

	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)
	
	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		conceptColumnsIndicesNext = conceptColumnsIndicesPred
		conceptColumnsFeatureIndicesNext = conceptColumnsFeatureIndicesPred
		kc = kcNetwork
		if(kc == 1 and kf == 1):
			multipleSourcesNext = False
		else:
			multipleSourcesNext = True
	else:
		#while exclusively training predictive network; use targets rather than next token predictions when activating database network
		conceptColumnsIndicesNext = targetConceptColumnsIndices
		conceptColumnsFeatureIndicesNext = targetConceptColumnsFeatureIndices
		#print("targetConceptColumnsIndices = ", targetConceptColumnsIndices)
		#print("targetConceptColumnsFeatureIndices = ", targetConceptColumnsFeatureIndices)
		multipleSourcesNext = targetMultipleSources
		if(multipleSourcesNext):
			kc = 2
		else:
			kc = 1
		assert kf==1
	conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)
	if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
		multipleSourcesNext = False
		kc = 0
	else:
		kc = conceptColumnsIndicesNext.shape[0]
		
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex
