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
printSequenceNumberColumns = True
modalityORRFfilterNamesVerbose = False	#default: False	#orig: True


#Execution mode selection;
if(useQuickExecution):
	executionMode = "train"
	inferenceTrainFirstSequences = False


#Dataset Type;
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
databaseFolderExtension = "-OR"


#Multisequence settings;
multisentencePredictions = False
numSentencesPerSequence = 1
spacyPipelineOptimisations = False
useBenchmarkDefaults = False


#Concept column delimiters;
usePOS = False
useDedicatedFeatureLists = False
conceptColumnsDelimitByPOS = False
pretrainConceptColumnsDelimitByPOSenforce = False
pretrainCombineConsecutiveNouns = False
pretrainCombineHyphenatedNouns = False
useSpacyForConceptNounPOSdetection = False
detectReferenceSetDelimitersBetweenNouns = False
trainConnectionStrengthPOSdependence = False
trainConnectionStrengthLimitTanh = False
trainConnectionStrengthLimitMax = False
inferenceConnectionStrengthPOSdependence = False


#modality OR;
modalityORframesPerSnapshot = 30
modalityORpixelsPerColumn = 20
modalityORnumberOfLayers = 5
modalityORtrainMaxLayerIndex = 0
modalityORuseExternalRFfilterLibrary = False
modalityORexternalRFfilterLibraryModuleName = "ATORpt_RF"
modalityORRFfilterThreshold = 0.2
modalityORsnapshotWidth = 160
modalityORsnapshotHeight = 90
modalityORvideoMinDurationSeconds = 60.0
modalityORvideoMaxDurationSeconds = 180.0
modalityORdatasetPromptRatio = 0.1
modalityORdatasetPromptMaxSequences = 2
