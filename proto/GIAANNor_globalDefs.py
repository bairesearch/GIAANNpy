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


#recent debug vars;
debugPrintNumberFeatures = True
debugPrintInsufficientUsableFeaturesWarning = True


#recent print vars;
printSequenceNumberColumns = True
modalityORRFfilterNamesVerbose = False	#default: False	#orig: True
modalityORsequenceDataTextSnapshotDelimiter = " | "
modalityORsequenceDataTextSegmentDelimiter = " \\ "
modalityORsequenceDataTextFeatureDelimiter = " "
modalityORsequenceDataTextSegmentPrefix = "s"
modalityORsequenceDataTextLabelSuffix = ": "
modalityORsequenceDataTextIndexDigits = 3


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
	datasetImageWidth = 1280
	datasetImageHeight = 720
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
	datasetImageWidth = 2048
	datasetImageHeight = 1024


#"layers" (regions) - could instantiate multiple GIAANN databases for independent/greedy training;
modalityORnumberOfLayers = 5
modalityORtrainMaxLayerIndex = 0


#RF Filters;
modalityORfiltersRotations = 18	#default: ~18 orientation columns per hypercolumn in V1
modalityORfiltersPerRotation = 1024	#debug: 16	#minimum: 1024	#default: ~15000 neuron per orientation column
modalityORRFfilterRotationallyInvariant = False	#orig: False
modalityORfilterChannels = modalityORfiltersPerRotation*modalityORfiltersRotations
if(modalityORRFfilterRotationallyInvariant):
	modalityORfilterChannelsOutput = modalityORfiltersPerRotation
else:
	modalityORfilterChannelsOutput = modalityORfiltersPerRotation*modalityORfiltersRotations
modalityORfilterUseExternalRFLibrary = False
modalityORfilterExternalRFLibraryModuleName = "ATORpt_RF"
modalityORfilterThreshold = 0.2
modalityORfilterTypeEllipsoidal = "ELL"
modalityORfilterTypeGabor = "GAB"
modalityORfilterTypePixel = "PIX"
modalityORfilterCodeOrientationPrefix = "_O"
modalityORfilterCodeVariantPrefix = "_V"
modalityORfilterCodeColourPrefix = "_C"
modalityORfilterCodePhasePrefix = "_P"
modalityORfilterCodeOutputPrefix = "OUT"
modalityORfilterCodeDefaultPrefix = "RF_"
modalityORfilterWordAlphabet = "abcdefghijklmnopqrstuvwxyz"
modalityORfilterCodeIndexDigits = 3
modalityORfilterCodeDefaultIndex = 0
modalityORfilterWordMinLetters = 2
modalityORfilterRadiansPerCircle = 6.283185307179586
modalityORfilterColourChannelCount = 3
modalityORfilterColourLuminance = (1.0, 1.0, 1.0)
modalityORfilterColourRedGreen = (1.0, -1.0, 0.0)
modalityORfilterColourGreenBlue = (0.0, 1.0, -1.0)
modalityORfilterColourRedBlue = (1.0, 0.0, -1.0)
modalityORfilterColourWeightsList = [modalityORfilterColourLuminance, modalityORfilterColourRedGreen, modalityORfilterColourGreenBlue, modalityORfilterColourRedBlue]
modalityORfilterColourIndexLuminance = 0
modalityORfilterColourIndexRedGreen = 1
modalityORfilterColourIndexGreenBlue = 2
modalityORfilterColourIndexRedBlue = 3
modalityORfilterPolarityPositive = 1.0
modalityORfilterPolarityNegative = -1.0
modalityORfilterPhaseCosine = 0.0
modalityORfilterPhaseSine = 1.5707963267948966
modalityORfilterFrequencyLow = 0.15
modalityORfilterFrequencyHigh = 0.25
modalityORfilterGeneratedTypeList = [modalityORfilterTypeEllipsoidal, modalityORfilterTypeGabor]
modalityORfilterGeneratedPolarityList = [modalityORfilterPolarityPositive, modalityORfilterPolarityNegative]
modalityORfilterGeneratedColourIndexList = [modalityORfilterColourIndexLuminance, modalityORfilterColourIndexRedGreen, modalityORfilterColourIndexGreenBlue, modalityORfilterColourIndexRedBlue]
modalityORfilterGeneratedFractionOffset = 0.5
modalityORfilterGeneratedMinimumFilterCount = 1
modalityORfilterGeneratedMinimumCategoryCount = 1
modalityORfilterGeneratedCoprimeStrideMinimum = 1
modalityORfilterGeneratedCoprimeSearchStep = 1
modalityORfilterGeneratedCoprimeGcdRequired = 1
modalityORfilterGeneratedTypePermutationStrideSeed = 7
modalityORfilterGeneratedTypePermutationOffsetSeed = 0
modalityORfilterGeneratedPolarityColourPermutationStrideSeed = 13
modalityORfilterGeneratedPolarityColourPermutationOffsetSeed = 1
modalityORfilterGeneratedFrequencyPermutationStrideSeed = 23
modalityORfilterGeneratedFrequencyPermutationOffsetSeed = 3
modalityORfilterGeneratedPhasePermutationStrideSeed = 29
modalityORfilterGeneratedPhasePermutationOffsetSeed = 5
modalityORfilterGeneratedSigmaXPermutationStrideSeed = 31
modalityORfilterGeneratedSigmaXPermutationOffsetSeed = 7
modalityORfilterGeneratedSigmaYPermutationStrideSeed = 37
modalityORfilterGeneratedSigmaYPermutationOffsetSeed = 11
modalityORfilterGeneratedLobeOffsetPermutationStrideSeed = 41
modalityORfilterGeneratedLobeOffsetPermutationOffsetSeed = 13
modalityORfilterGeneratedSurroundScalePermutationStrideSeed = 43
modalityORfilterGeneratedSurroundScalePermutationOffsetSeed = 17
modalityORfilterGeneratedFrequencyMin = 0.05
modalityORfilterGeneratedFrequencyMax = 0.45
modalityORfilterGeneratedPhaseMin = 0.0
modalityORfilterGeneratedPhaseMax = modalityORfilterRadiansPerCircle
modalityORfilterGeneratedSigmaXMin = 0.2
modalityORfilterGeneratedSigmaXMax = 0.7
modalityORfilterGeneratedSigmaYMin = 0.08
modalityORfilterGeneratedSigmaYMax = 0.35
modalityORfilterGeneratedLobeOffsetMin = 0.2
modalityORfilterGeneratedLobeOffsetMax = 0.7
modalityORfilterGeneratedSurroundScaleMin = 0.25
modalityORfilterGeneratedSurroundScaleMax = 0.75
modalityORfilterEllipsoidalSigmaX = 0.45
modalityORfilterEllipsoidalSigmaY = 0.2
modalityORfilterEllipsoidalLobeOffset = 0.4
modalityORfilterEllipsoidalLobeSigmaXScale = 1.2
modalityORfilterEllipsoidalLobeSigmaYScale = 1.4
modalityORfilterEllipsoidalSurroundScale = 0.5
modalityORfilterGaborSigmaX = 0.45
modalityORfilterGaborSigmaY = 0.25
modalityORfilterSupplementaryOrientationCount = 8
modalityORfilterSupplementaryRadiusCount = 2
modalityORfilterSupplementaryRadius = 0.35
modalityORfilterSupplementarySigmaBase = 0.2
modalityORfilterSupplementarySigmaStep = 0.05
modalityORfilterSupplementarySigmaCount = 3
modalityORfilterSupplementarySurroundSigmaScale = 2.0
modalityORfilterSupplementarySurroundScale = 0.5
modalityORfilterPrototypeTypeIndex = 0
modalityORfilterPrototypePolarityIndex = 1
modalityORfilterPrototypeFrequencyIndex = 2
modalityORfilterPrototypePhaseIndex = 3
modalityORfilterPrototypeColourIndex = 4
modalityORfilterPrototypeSigmaXIndex = 5
modalityORfilterPrototypeSigmaYIndex = 6
modalityORfilterPrototypeLobeOffsetIndex = 7
modalityORfilterPrototypeSurroundScaleIndex = 8
modalityORfilterPrototypeLength = 9
modalityORfilterNoFrequency = 0.0
modalityORfilterNoLobeOffset = 0.0
modalityORfilterNoSurroundScale = 1.0


#snapshots (subimage extraction via saccades);
modalityORsnapshotFractionOfImage = 0.25
modalityORsnapshotRetinotopicFieldBias = True
if(modalityORsnapshotRetinotopicFieldBias):
	modalityORsnapshotRetinotopicFieldMaxDegrees = modalityORsnapshotFractionOfImage*datasetCameraFOV
else:
	modalityORsnapshotRetinotopicFieldMaxDegrees = None


#VX column/feature assignment;
tokensiationMethodOneColumnPerSnapshotPixel = True	#default: True #orig: False
if(tokensiationMethodOneColumnPerSnapshotPixel):
	modalityORnumberOfColumnsVX = 1000	#default: 1000	#~1000 hypercolumns in V1
	modalityORfilterWidth = 5	#default: 5
	modalityORnumberOfFeaturesPerColumn = modalityORfilterWidth*modalityORfilterWidth*modalityORfilterChannelsOutput
else:
	modalityORpixelsPerColumn = 20
	modalityORsnapshotDimension = int(round(float(min(datasetImageWidth, datasetImageHeight))*float(modalityORsnapshotFractionOfImage)))
	if(modalityORsnapshotDimension < modalityORpixelsPerColumn):
		raise RuntimeError("GIAANNor_globalDefs error: modalityORsnapshotDimension must be >= modalityORpixelsPerColumn")
	modalityORnumberOfColumnsVXaxis = int(modalityORsnapshotDimension//modalityORpixelsPerColumn)
	if(modalityORnumberOfColumnsVXaxis <= 0):
		raise RuntimeError("GIAANNor_globalDefs error: modalityORnumberOfColumnsVXaxis must be > 0")
	modalityORnumberOfColumnsVX = modalityORnumberOfColumnsVXaxis*modalityORnumberOfColumnsVXaxis


#submodality settings;
if(submodalityName=="video"):
	modalityORvideoGenerateMultipleSnapshotsPerFrame = True	#default: True #orig: False	#modality OR video generate multiple snapshots per frame using detected keypoints
	modalityORvideoGenerateMultipleSnapshotsPerFrameParallel = True
	modalityORvideoFrameRate = 30
	modalityORvideoFramesPerSequenceIteration = 30
	modalityORvideoMinDurationSeconds = 60.0
	modalityORvideoMaxDurationSeconds = 180.0
	modalityORdatasetPromptRatio = 0.1
	modalityORdatasetPromptMaxSequences = 2
	modalityORvideoMaxEncodedSnapshotsPerSequence = 10	#max value:  int((float(modalityORvideoMaxDurationSeconds)*float(modalityORvideoFrameRate))/float(modalityORvideoFramesPerSequenceIteration))
	modalityORvideoStreamFrames = False	#orig: False
elif(submodalityName=="image"):
	#saccade augmentations are calculated by translating the image to a polar coordinates offset from the centre
	modalityORimageMaxSequencesPerImage = 5		#max independent sequences per image (lists of saccade keypoints)
	modalityORimageSequenceEncode = "axes"	#orig: "saccades"	#options: "saccades", "distance", "axis", "axes", "none"
	if(modalityORimageSequenceEncode=="axes"):
		modalityORimageSequenceEncodeAxesColumnRandom = True
		modalityORimageSequenceEncodeAxesSourceColumnIndex = 0	#mandatory: 0	#encoded source column index used by modalityORimageSequenceEncode=="axes"
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			modalityORnumberOfColumnsVS = 1000	#VS: Visual Spectral area (hypothetical)
		else:
			modalityORnumberOfColumnsVS = 1
			modalityORimageSequenceEncodeAxesTargetColumnIndex = 0	#mandatory: 0	#fixed encoded target column index used by modalityORimageSequenceEncode=="axes"
	if(modalityORimageSequenceEncode=="saccades"):
		modalityORimageSaccadeKeypointsPerEncoding = 3	#default: 3	#orig: 2	# the number of saccade keypoints (transition points) used to encode the context of each selected column feature neuron
		modalityORimageSnapshotsPerSaccade = 1	#default: 1	#if 1: one snapshot per saccade (plus the start point), if >1 add interpolation between saccade keypoints.
		modalityORimageSnapshotsPerSequence = modalityORimageSaccadeKeypointsPerEncoding*modalityORimageSnapshotsPerSaccade+1
	elif(modalityORimageSequenceEncode=="distance" or modalityORimageSequenceEncode=="axis" or modalityORimageSequenceEncode=="axes"):
		modalityORimageSequenceEncodeDistanceFieldSegments = 8	#total number of segments to divide the snapshot visual field into (x or y)
		modalityORimageSaccadeKeypointsPerEncoding = 1	#mandatory: 1
		modalityORimageSnapshotsPerSaccade = 0 #mandatory: 0	#if 0: no temporal encoding of saccade movements
		modalityORimageSnapshotsPerSequence = 1
	elif(modalityORimageSequenceEncode=="none"):
		modalityORimageSaccadeKeypointsPerEncoding = 1	#mandatory: 1
		modalityORimageSnapshotsPerSaccade = 0 #mandatory: 0	#if 0: no temporal encoding of saccade movements
		modalityORimageSnapshotsPerSequence = 1
	else:
		raise RuntimeError("GIAANNor_globalDefs error: modalityORimageSequenceEncode must be 'saccades', 'distance', 'axis', 'axes', or 'none'")
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


#feature detection (for keypoints) - this preprocessing is implicitly assumed performed in VX via RFs of different shapes and sizes;
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
