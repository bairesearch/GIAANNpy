"""GIAANNor_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_globalDefs.py

# Usage:
see GIAANNcmn_globalDefs.py

# Description:
GIA ANN OR global Defs

"""

from GIAANNcmn_globalDefs import useQuickExecution
from GIAANNcmn_globalDefs import useBenchmark
from GIAANNcmn_globalDefs import useAutoresearch
from GIAANNcmn_globalDefs import useDrawNetworkIndependently


#recent print vars;
#submodalityName = "video"	#default
submodalityName = "image"

#recent print vars;
printSequenceNumberColumns = True
modalityORRFfilterNamesVerbose = False	#default: False	#orig: True


#Dataset Type;
if(submodalityName=="video"):
	datasetType = "soccer_events"
	datasetName = "infactory-ai/soccer-events"
	datasetCfg = ""
	datasetsLibrary4plus = True
	useLocalDataset = False
	useLocalDatasetDownloadManual = False
	datasetFolder = "../../dataset/or/"
	datasetProcessedCacheFolder = ""
	trainLoadExistingDatabase = True
	trainTestSet = False
	generateEvalText = False
	trainSetStartOffsetSequences = 0
	inferencePromptFileName = "inference_prompt_or.txt"
	databaseFolderExtension = "-ORvideo"
elif(submodalityName=="image"):
	datasetType = "cifar10"
	datasetName = "uoft-cs/cifar10"
	datasetCfg = "plain_text"
	datasetsLibrary4plus = True
	useLocalDataset = False
	useLocalDatasetDownloadManual = False
	datasetFolder = "../../dataset/or/"
	datasetProcessedCacheFolder = ""
	trainLoadExistingDatabase = True
	trainTestSet = False
	generateEvalText = False
	trainSetStartOffsetSequences = 0
	inferencePromptFileName = "inference_prompt_or.txt"
	databaseFolderExtension = "-ORimage"


#unused nlp settings;
	#Multisequence settings;
multisentencePredictions = False
numSentencesPerSequence = 1
spacyPipelineOptimisations = False
useBenchmarkDefaults = False
	#Concept column delimiters;
usePOS = False
useDedicatedFeatureLists = False
conceptColumnsDelimitByPOS = False
predictionColumnsMustActivateConceptFeature = False
pretrainCombineConsecutiveNouns = False
pretrainCombineHyphenatedNouns = False
pretrainConceptColumnsDelimitByPOSenforce = False
useSpacyForConceptNounPOSdetection = False
detectReferenceSetDelimitersBetweenNouns = False
trainConnectionStrengthPOSdependence = False
trainConnectionStrengthLimitTanh = False
trainConnectionStrengthLimitMax = False
inferenceConnectionStrengthPOSdependence = False


#modality OR;
modalityORpixelsPerColumn = 20
modalityORnumberOfLayers = 5
modalityORtrainMaxLayerIndex = 0
modalityORuseExternalRFfilterLibrary = False
modalityORexternalRFfilterLibraryModuleName = "ATORpt_RF"
modalityORRFfilterThreshold = 0.2
modalityORsnapshotWidth = 160
modalityORsnapshotHeight = 90

#modality OR video:
if(submodalityName=="video"):
	modalityORvideoFramesPerSnapshot = 30
	modalityORvideoMinDurationSeconds = 60.0
	modalityORvideoMaxDurationSeconds = 180.0
	modalityORdatasetPromptRatio = 0.1
	modalityORdatasetPromptMaxSequences = 2
elif(submodalityName=="image"):
	#saccade augmentations are calculated by translating the image to a random polar coordinates offset from the centre
	modalityORimageSaccadesMaxAngularOffsetDegrees = 15	#effective augmentations compatible with SANI
	modalityORimageSaccadesPerImage = 5
	modalityORimageSnapshotsPerSaccade = 3
	modalityORdatasetPromptRatio = 0.1
	modalityORdatasetPromptMaxSequences = 2
