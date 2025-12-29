"""GIAANNproto_databaseNetworkFilesInhibition.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Files Inhibition

"""

import os
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_databaseNetworkInhibition

inhibitoryObservedColumnsDir = os.path.join(databaseFolder, "observedColumnsInhibitory")

def loadInhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma):
	featureNeuronsPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureNeurons{pytorchTensorFileExtension}")
	featureConnectionsOutputPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureConnectionsOutput{pytorchTensorFileExtension}")
	featureConnectionsInputPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureConnectionsInput{pytorchTensorFileExtension}")
	legacyConnectionsPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureConnections{pytorchTensorFileExtension}")
	if os.path.exists(featureNeuronsPath) and (os.path.exists(featureConnectionsOutputPath) or os.path.exists(legacyConnectionsPath)):
		featureNeurons = pt.load(featureNeuronsPath).to(deviceSparse)
		if(os.path.exists(featureConnectionsOutputPath)):
			featureConnectionsOutput = pt.load(featureConnectionsOutputPath).to(deviceSparse)
		else:
			featureConnectionsOutput = pt.load(legacyConnectionsPath).to(deviceSparse)
		if(os.path.exists(featureConnectionsInputPath)):
			featureConnectionsInput = pt.load(featureConnectionsInputPath).to(deviceSparse)
		else:
			featureConnectionsInput = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, databaseNetworkObject.f, databaseNetworkObject.c, databaseNetworkObject.f))
		column = GIAANNproto_databaseNetworkInhibition.InhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma)
		column.featureNeurons = featureNeurons
		column.featureConnectionsOutput = featureConnectionsOutput
		column.featureConnectionsInput = featureConnectionsInput
		column.resizeConceptArrays(databaseNetworkObject.c)
		column.expandFeatureArrays(databaseNetworkObject.f)
	else:
		column = GIAANNproto_databaseNetworkInhibition.InhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma)
	return column

def saveObservedColumnInhibition(self):
	os.makedirs(inhibitoryObservedColumnsDir, exist_ok=True)
	featureNeuronsPath = os.path.join(inhibitoryObservedColumnsDir, f"{self.conceptIndex}_inhib_featureNeurons{pytorchTensorFileExtension}")
	featureConnectionsOutputPath = os.path.join(inhibitoryObservedColumnsDir, f"{self.conceptIndex}_inhib_featureConnectionsOutput{pytorchTensorFileExtension}")
	featureConnectionsInputPath = os.path.join(inhibitoryObservedColumnsDir, f"{self.conceptIndex}_inhib_featureConnectionsInput{pytorchTensorFileExtension}")
	pt.save(self.featureNeurons.coalesce(), featureNeuronsPath)
	pt.save(self.featureConnectionsOutput.coalesce(), featureConnectionsOutputPath)
	pt.save(self.featureConnectionsInput.coalesce(), featureConnectionsInputPath)

def getInhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma):
	if not hasattr(databaseNetworkObject, "inhibitoryObservedColumnsDict"):
		databaseNetworkObject.inhibitoryObservedColumnsDict = {}
	columnsDict = databaseNetworkObject.inhibitoryObservedColumnsDict
	if conceptIndex in columnsDict:
		columnObject = columnsDict[conceptIndex]
		columnObject.resizeConceptArrays(databaseNetworkObject.c)
		columnObject.expandFeatureArrays(databaseNetworkObject.f)
		return columnObject
	column = loadInhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma)
	columnsDict[conceptIndex] = column
	return column

