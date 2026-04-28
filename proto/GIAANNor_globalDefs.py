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


#submodality selection;
submodalityName = "image"	#image, video


#algorithm selection
tokensiationMethodOneColumnPerSnapshotPixel = True	#default: True #orig: False


#recent debug vars;
debugPrintNumberFeatures = True
debugPrintInsufficientUsableFeaturesWarning = True


#recent print vars;
printSequenceNumberColumns = True
modalityORRFfilterNamesVerbose = False	#default: False	#orig: True


#Dataset Type;
if(submodalityName=="video"):
	datasetType = "soccer_events"
	datasetName = "infactory-ai/soccer-events"
	datasetCfg = ""
	datasetsLibrary4plus = True
	useLocalDataset = True
	useLocalDatasetDownloadManual = False
	datasetFolder = "../../dataset/or/"
	datasetProcessedCacheFolder = ""
	trainLoadExistingDatabase = True
	trainTestSet = False
	generateEvalText = False
	trainSetStartOffsetSequences = 0
	inferencePromptFileName = "inference_prompt_or.txt"
	databaseFolderExtension = "-ORvideo"
	modalityORdatasetHasBackground = False
	datasetCameraHorizontalFOV = 50.0	#estimated dataset-level FOV for infactory-ai/soccer-events 1280x720 professional soccer broadcast clips; source footage is variable-zoom and has no published per-clip intrinsics.
	datasetCameraFOV = datasetCameraHorizontalFOV/2.0	#retinotopic table uses eccentricity/radius from fixation, not full horizontal FOV
elif(submodalityName=="image"):
	datasetType = "cityscapes"
	datasetName = "Chris1/cityscapes"
	datasetCfg = ""
	datasetsLibrary4plus = True
	useLocalDataset = True
	useLocalDatasetDownloadManual = False
	datasetFolder = "../../dataset/or/"
	datasetProcessedCacheFolder = ""
	trainLoadExistingDatabase = True
	trainTestSet = False
	generateEvalText = False
	trainSetStartOffsetSequences = 0
	inferencePromptFileName = "inference_prompt_or.txt"
	databaseFolderExtension = "-ORimage"
	modalityORdatasetHasBackground = False
	datasetCameraHorizontalFOV = 50.0	#48.70231915612077	#derived from Cityscapes K: fx=2262.52, width=2048; 2*atan(width/(2*fx))
	datasetCameraFOV = datasetCameraHorizontalFOV/2.0	#retinotopic table uses eccentricity/radius from fixation, not full horizontal FOV

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
if(tokensiationMethodOneColumnPerSnapshotPixel):
	modalityORnumberOfColumns = 1000
	modalityORfilterWidth = 5
	modalityORfilterChannels = 100
	modalityORnumberOfFeaturesPerColumn = modalityORfilterWidth*modalityORfilterWidth*modalityORfilterChannels
else:
	modalityORpixelsPerColumn = 20
modalityORnumberOfLayers = 5
modalityORtrainMaxLayerIndex = 0
modalityORuseExternalRFfilterLibrary = False
modalityORexternalRFfilterLibraryModuleName = "ATORpt_RF"
modalityORRFfilterThreshold = 0.2
modalityORsnapshotFractionOfImage = 0.25
modalityORsnapshotRetinotopicFieldBias = True
if(modalityORsnapshotRetinotopicFieldBias):
	modalityORsnapshotRetinotopicFieldMaxDegrees = modalityORsnapshotFractionOfImage*datasetCameraFOV
else:
	modalityORsnapshotRetinotopicFieldMaxDegrees = None

if(submodalityName=="video"):
	modalityORvideoGenerateMultipleSnapshotsPerFrame = True	#default: True #orig: False	#modality OR video generate multiple snapshots per frame using detected keypoints
	modalityORvideoGenerateMultipleSnapshotsPerFrameParallel = True
	modalityORvideoFrameRate = 30
	modalityORvideoFramesPerSequenceIteration = 30
	modalityORvideoMinDurationSeconds = 60.0
	modalityORvideoMaxDurationSeconds = 180.0
	modalityORdatasetPromptRatio = 0.1
	modalityORdatasetPromptMaxSequences = 2
	modalityORvideoStreamFrames = False	#orig: False
elif(submodalityName=="image"):
	modalityORimageSaccadesEncode = False	#default: False	#orig: True
	#saccade augmentations are calculated by translating the image to a random polar coordinates offset from the centre
	modalityORimageSaccadesPerImage = 5		#max number saccades per image
	if(modalityORimageSaccadesEncode):
		modalityORimageSnapshotsPerSaccade = 2	#default: 2	#if 2: start and end of saccade, if >2 start and end of saccade and modalityORimageSnapshotsPerSaccade-2 interpolations between these.
	else:
		modalityORimageSnapshotsPerSaccade = 1 #mandatory: 1	#if 1: no temporal encoding of saccade movements
	# upgrade submodalityName=="image" to perform saccades augomentations between nearby (i.e. adjacent) salient regions of the image: a) segment centres and b) corner features
	modalityORimageSaccadesUseAdjacentSalientRegions = True
	modalityORimageSaccadesSkipInsufficientUsableFeatures = True
	if modalityORimageSaccadesUseAdjacentSalientRegions:
		modalityORimageSaccadesCrop = False
		modalityORimageSaccadesNumberOfNearbyPairs = 4	#default: 3	#orig: 2
	else:
		modalityORimageSaccadesMaxAngularOffsetDegrees = 15	#effective augmentations compatible with SANI
		modalityORimageSaccadesCrop = False	#uses modalityORimageSaccadesMaxAngularOffsetDegrees to determine crop distance (ensures to not produce augmentations with blank areas)
	modalityORdatasetPromptRatio = 0.1
	modalityORdatasetPromptMaxSequences = 2

#feature detection (for keypoints);
modalityORfeatureDetectionCorners = False	#default: False
modalityORfeatureDetectionSegmentCentres = True	#default: True
modalityORfeatureDetectionSegmentPostProcessing = True
modalityORfeatureDetectionSegmentMetadata = True
modalityORfeatureDetectionFilterSegments = True
if(modalityORdatasetHasBackground):
	modalityORfeatureDetectionFilterSegmentsWholeImageThreshold = 0.85	#requires adjusting for dataset
	modalityORfeatureDetectionFilterSegmentsBackgroundColourThreshold = 15	#requires adjusting for dataset
else:
	modalityORfeatureDetectionFilterSegmentsWholeImageThreshold = 1.0	#set modalityORfeatureDetectionFilterSegmentsWholeImageThreshold=1.0 to reject whole-image segments (for image datasets that do not have a background colour)
	modalityORfeatureDetectionFilterSegmentsBackgroundColourThreshold = 0
modalityORfeatureDetectionSAMversion = 2	#1=sam1(segment-anything), 2=sam2, 3=sam3
modalityORfeatureDetectionSAM1modelName = "vit_h"
modalityORfeatureDetectionSAM1checkpoint = "../../models/segmentAnythingViTHSAM/sam_vit_h_4b8939.pth"
modalityORfeatureDetectionSAM1checkpointAutoDownload = True
modalityORfeatureDetectionSAM2modelId = "facebook/sam2.1-hiera-large"
modalityORfeatureDetectionSAM2configFile = "configs/sam2.1/sam2.1_hiera_l.yaml"
modalityORfeatureDetectionSAM2checkpoint = ""
modalityORfeatureDetectionSAM3checkpoint = ""
modalityORfeatureDetectionSAM3textPrompt = "object"
modalityORfeatureDetectionSAM3confidenceThreshold = 0.5
