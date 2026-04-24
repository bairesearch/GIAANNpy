"""GIAANNor_features.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR features

"""

import os
import shutil
import urllib.request
import torch as pt

from GIAANNcmn_globalDefs import *

_segmentAnythingRuntimeCache = None

xAxisFeatureMap = 0
yAxisFeatureMap = 1


def requireOpenCV():
	result = None
	try:
		import cv2 as cv2Import
		result = cv2Import
	except Exception as exception:
		raise RuntimeError("requireOpenCV error: missing cv2 (pip install opencv-python)") from exception
	return result


def requireNumPy():
	result = None
	try:
		import numpy as npImport
		result = npImport
	except Exception as exception:
		raise RuntimeError("requireNumPy error: missing numpy (pip install numpy)") from exception
	return result


def requireSAM1():
	result = None
	SamAutomaticMaskGenerator = None
	sam_model_registry = None
	try:
		from segment_anything import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorImport, sam_model_registry as sam_model_registryImport
		SamAutomaticMaskGenerator = SamAutomaticMaskGeneratorImport
		sam_model_registry = sam_model_registryImport
	except Exception as exception:
		raise RuntimeError("requireSAM1 error: missing segment_anything (pip install git+https://github.com/facebookresearch/segment-anything.git)") from exception
	result = (SamAutomaticMaskGenerator, sam_model_registry)
	return result


def requireSAM2():
	result = None
	SAM2AutomaticMaskGenerator = None
	build_sam2 = None
	try:
		from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as SAM2AutomaticMaskGeneratorImport
		from sam2.build_sam import build_sam2 as build_sam2Import
		SAM2AutomaticMaskGenerator = SAM2AutomaticMaskGeneratorImport
		build_sam2 = build_sam2Import
	except Exception as exception:
		raise RuntimeError("requireSAM2 error: missing sam2 (pip install git+https://github.com/facebookresearch/sam2.git)") from exception
	result = (SAM2AutomaticMaskGenerator, build_sam2)
	return result


def requireSAM3():
	result = None
	build_sam3_image_model = None
	Sam3Processor = None
	try:
		from sam3 import build_sam3_image_model as build_sam3_image_modelImport
		from sam3.model.sam3_image_processor import Sam3Processor as Sam3ProcessorImport
		build_sam3_image_model = build_sam3_image_modelImport
		Sam3Processor = Sam3ProcessorImport
	except Exception as exception:
		raise RuntimeError("requireSAM3 error: missing sam3 (pip install git+https://github.com/facebookresearch/sam3.git)") from exception
	result = (build_sam3_image_model, Sam3Processor)
	return result


def createEmptyFeatureCoordinatesTensor():
	result = pt.zeros((0, 2), dtype=pt.float32, device=deviceDense)
	return result


def detectSalientFeatureCoordinatesFromImage(imageSource, zoomIndex=0):
	# create GIAANNor_features.py - copy the ATORpt feature detection code that uses the Segment Anything library and OpenCV.
	result = None
	image = None
	featureCoordinates = None
	sanitisedFeatureCoordinates = None
	image = loadImageRGB(imageSource)
	featureCoordinates = featureDetection(image, zoomIndex)
	if(featureCoordinates.shape[0] > 0):
		sanitisedFeatureCoordinates = sanitiseFeatureCoordinates(featureCoordinates, image.shape[1], image.shape[0])
	else:
		sanitisedFeatureCoordinates = createEmptyFeatureCoordinatesTensor()
	if(debugPrintNumberFeatures):
		printSanitisedFeatureDetectionCount(sanitisedFeatureCoordinates)
	result = sanitisedFeatureCoordinates
	return result


def detectSalientFeatureCoordinatesFromImageTensor(imageTensor):
	# create GIAANNor_features.py - copy the ATORpt feature detection code that uses the Segment Anything library and OpenCV.
	result = None
	image = None
	image = convertImageTensorToNumpyRGB(imageTensor)
	result = detectSalientFeatureCoordinatesFromImage(image, 0)
	return result


def detectSegmentAnythingFeaturesFromImage(imageSource):
	result = None
	image = None
	image = loadImageRGB(imageSource)
	result = detectSegmentAnythingFeatures(image)
	return result


def featureDetection(image, zoomIndex):
	result = None
	zoom = None
	imageFeatureCoordinates = None
	cornerFeatureCoordinates = None
	segmentFeatureCoordinates = None
	validateRGBImage(image, "featureDetection")
	zoom = getZoomValue(zoomIndex)
	imageFeatureCoordinates = pt.zeros((0, 2), dtype=pt.float32, device=deviceDense)
	cornerFeatureCoordinates = pt.zeros((0, 2), dtype=pt.float32, device=deviceDense)
	segmentFeatureCoordinates = pt.zeros((0, 2), dtype=pt.float32, device=deviceDense)
	if(modalityORimageFeatureDetectionCorners):
		cornerFeatureCoordinates = featureDetectionCornerOpenCVHarris(image)
		imageFeatureCoordinates = pt.cat((imageFeatureCoordinates, cornerFeatureCoordinates), dim=0)
	if(modalityORimageFeatureDetectionSegmentCentres):
		segmentFeatureCoordinates = featureDetectionCentroidFBSegmentAnything(image)
		imageFeatureCoordinates = pt.cat((imageFeatureCoordinates, segmentFeatureCoordinates), dim=0)
	if(debugPrintNumberFeatures):
		printFeatureDetectionCounts(cornerFeatureCoordinates, segmentFeatureCoordinates)
	result = imageFeatureCoordinates*zoom
	return result


def getZoomValue(zoomIndex):
	# sync with ATORmethodClass::createOrAddPointsToFeaturesList
	result = None
	if(not isinstance(zoomIndex, int)):
		raise RuntimeError("getZoomValue error: zoomIndex must be an int")
	if(zoomIndex < 0):
		raise RuntimeError("getZoomValue error: zoomIndex must be >= 0")
	result = int(pow(2, zoomIndex))
	return result


def requireSegmentAnythingRuntime():
	result = None
	global _segmentAnythingRuntimeCache
	if(_segmentAnythingRuntimeCache is None):
		if(modalityORfeatureDetectionSAMversion == 1):
			_segmentAnythingRuntimeCache = createSAM1maskGenerator()
		elif(modalityORfeatureDetectionSAMversion == 2):
			_segmentAnythingRuntimeCache = createSAM2maskGenerator()
		elif(modalityORfeatureDetectionSAMversion == 3):
			_segmentAnythingRuntimeCache = createSAM3processor()
		else:
			raise RuntimeError("requireSegmentAnythingRuntime error: unsupported modalityORfeatureDetectionSAMversion " + str(modalityORfeatureDetectionSAMversion))
	result = _segmentAnythingRuntimeCache
	return result


def createSAM1maskGenerator():
	result = None
	SamAutomaticMaskGenerator = None
	sam_model_registry = None
	sam = None
	samDevice = None
	samCheckpointPath = None
	SamAutomaticMaskGenerator, sam_model_registry = requireSAM1()
	if(modalityORfeatureDetectionSAM1modelName not in sam_model_registry):
		raise RuntimeError("createSAM1maskGenerator error: unsupported SAM1 model name " + str(modalityORfeatureDetectionSAM1modelName))
	if(modalityORfeatureDetectionSAM1checkpointAutoDownload):
		samCheckpointPath = resolveSAM1checkpointPath()
	else:
		if(modalityORfeatureDetectionSAM1checkpoint == ""):
			raise RuntimeError("createSAM1maskGenerator error: modalityORfeatureDetectionSAM1checkpoint must not be empty")
		if(not os.path.isfile(modalityORfeatureDetectionSAM1checkpoint)):
			raise RuntimeError("createSAM1maskGenerator error: missing SAM1 checkpoint " + str(modalityORfeatureDetectionSAM1checkpoint))
		samCheckpointPath = modalityORfeatureDetectionSAM1checkpoint
	sam = sam_model_registry[modalityORfeatureDetectionSAM1modelName](checkpoint=samCheckpointPath)
	samDevice = deviceDense.type
	sam = sam.to(device=samDevice)
	result = SamAutomaticMaskGenerator(sam)
	return result


def resolveSAM1checkpointPath():
	result = None
	checkpointPath = None
	if(modalityORfeatureDetectionSAM1checkpoint != ""):
		checkpointPath = modalityORfeatureDetectionSAM1checkpoint
	else:
		checkpointPath = getDefaultSAM1checkpointPath()
	if(not os.path.isfile(checkpointPath)):
		downloadSAM1checkpoint(checkpointPath)
	if(not os.path.isfile(checkpointPath)):
		raise RuntimeError("resolveSAM1checkpointPath error: missing SAM1 checkpoint after attempted download " + str(checkpointPath))
	result = checkpointPath
	return result


def getDefaultSAM1checkpointPath():
	result = None
	checkpointFilename = None
	checkpointDirectory = None
	checkpointFilename = getSAM1checkpointFilename()
	checkpointDirectory = "../../models/segmentAnything"
	result = os.path.join(checkpointDirectory, checkpointFilename)
	return result


def getSAM1checkpointFilename():
	result = None
	if(modalityORfeatureDetectionSAM1modelName == "default" or modalityORfeatureDetectionSAM1modelName == "vit_h"):
		result = "sam_vit_h_4b8939.pth"
	elif(modalityORfeatureDetectionSAM1modelName == "vit_l"):
		result = "sam_vit_l_0b3195.pth"
	elif(modalityORfeatureDetectionSAM1modelName == "vit_b"):
		result = "sam_vit_b_01ec64.pth"
	else:
		raise RuntimeError("getSAM1checkpointFilename error: unsupported SAM1 model name " + str(modalityORfeatureDetectionSAM1modelName))
	return result


def getSAM1checkpointURL():
	result = None
	checkpointFilename = None
	checkpointFilename = getSAM1checkpointFilename()
	result = "https://dl.fbaipublicfiles.com/segment_anything/" + checkpointFilename
	return result


def downloadSAM1checkpoint(checkpointPath):
	result = None
	checkpointDirectory = None
	checkpointURL = None
	temporaryCheckpointPath = None
	response = None
	fileObject = None
	if(checkpointPath == ""):
		raise RuntimeError("downloadSAM1checkpoint error: checkpointPath must not be empty")
	checkpointDirectory = os.path.dirname(checkpointPath)
	if(checkpointDirectory != ""):
		os.makedirs(checkpointDirectory, exist_ok=True)
	checkpointURL = getSAM1checkpointURL()
	temporaryCheckpointPath = checkpointPath + ".tmp"
	try:
		response = urllib.request.urlopen(checkpointURL)
		fileObject = open(temporaryCheckpointPath, "wb")
		shutil.copyfileobj(response, fileObject)
	except Exception as exception:
		if(fileObject is not None):
			fileObject.close()
		if(response is not None):
			response.close()
		if(os.path.exists(temporaryCheckpointPath)):
			os.remove(temporaryCheckpointPath)
		raise RuntimeError("downloadSAM1checkpoint error: failed to download SAM1 checkpoint from " + str(checkpointURL) + " to " + str(checkpointPath)) from exception
	if(fileObject is not None):
		fileObject.close()
	if(response is not None):
		response.close()
	os.replace(temporaryCheckpointPath, checkpointPath)
	result = checkpointPath
	return result


def createSAM2maskGenerator():
	result = None
	SAM2AutomaticMaskGenerator = None
	build_sam2 = None
	model = None
	SAM2AutomaticMaskGenerator, build_sam2 = requireSAM2()
	if(modalityORfeatureDetectionSAM2checkpoint != ""):
		if(modalityORfeatureDetectionSAM2configFile == ""):
			raise RuntimeError("createSAM2maskGenerator error: modalityORfeatureDetectionSAM2configFile must not be empty when modalityORfeatureDetectionSAM2checkpoint is set")
		if(not os.path.isfile(modalityORfeatureDetectionSAM2checkpoint)):
			raise RuntimeError("createSAM2maskGenerator error: missing SAM2 checkpoint " + str(modalityORfeatureDetectionSAM2checkpoint))
		model = build_sam2(modalityORfeatureDetectionSAM2configFile, modalityORfeatureDetectionSAM2checkpoint, device=deviceDense.type)
		result = SAM2AutomaticMaskGenerator(model)
	else:
		if(modalityORfeatureDetectionSAM2modelId == ""):
			raise RuntimeError("createSAM2maskGenerator error: modalityORfeatureDetectionSAM2modelId must not be empty")
		result = SAM2AutomaticMaskGenerator.from_pretrained(modalityORfeatureDetectionSAM2modelId, device=deviceDense.type)
	return result


def createSAM3processor():
	result = None
	build_sam3_image_model = None
	Sam3Processor = None
	model = None
	if(deviceDense.type != "cuda"):
		raise RuntimeError("createSAM3processor error: SAM3 requires a CUDA-compatible GPU according to the official installation prerequisites")
	if(modalityORfeatureDetectionSAM3confidenceThreshold < 0.0 or modalityORfeatureDetectionSAM3confidenceThreshold > 1.0):
		raise RuntimeError("createSAM3processor error: modalityORfeatureDetectionSAM3confidenceThreshold must be >= 0.0 and <= 1.0")
	build_sam3_image_model, Sam3Processor = requireSAM3()
	if(modalityORfeatureDetectionSAM3checkpoint != ""):
		if(not os.path.isfile(modalityORfeatureDetectionSAM3checkpoint)):
			raise RuntimeError("createSAM3processor error: missing SAM3 checkpoint " + str(modalityORfeatureDetectionSAM3checkpoint))
		model = build_sam3_image_model(device=deviceDense.type, checkpoint_path=modalityORfeatureDetectionSAM3checkpoint, load_from_HF=False)
	else:
		model = build_sam3_image_model(device=deviceDense.type, checkpoint_path=None, load_from_HF=True)
	result = Sam3Processor(model, device=deviceDense.type, confidence_threshold=modalityORfeatureDetectionSAM3confidenceThreshold)
	return result


def generateSegmentAnythingMasks(image):
	result = None
	segmentAnythingRuntime = None
	segmentAnythingRuntime = requireSegmentAnythingRuntime()
	if(modalityORfeatureDetectionSAMversion == 3):
		result = generateSAM3masks(segmentAnythingRuntime, image)
	else:
		result = segmentAnythingRuntime.generate(image)
	return result


def generateSAM3masks(sam3Processor, image):
	result = []
	state = None
	sam3Masks = None
	maskIndex = None
	mask = None
	if(modalityORfeatureDetectionSAM3textPrompt == ""):
		raise RuntimeError("generateSAM3masks error: modalityORfeatureDetectionSAM3textPrompt must not be empty")
	state = sam3Processor.set_image(image)
	state = sam3Processor.set_text_prompt(modalityORfeatureDetectionSAM3textPrompt, state)
	if("masks" not in state):
		raise RuntimeError("generateSAM3masks error: SAM3 state missing masks")
	sam3Masks = state["masks"]
	if(not pt.is_tensor(sam3Masks)):
		raise RuntimeError("generateSAM3masks error: SAM3 masks must be a tensor")
	if(sam3Masks.dim() == 4):
		if(int(sam3Masks.shape[1]) != 1):
			raise RuntimeError("generateSAM3masks error: SAM3 masks rank 4 tensors must have channel dimension 1")
		sam3Masks = sam3Masks[:, 0]
	elif(sam3Masks.dim() != 3):
		raise RuntimeError("generateSAM3masks error: SAM3 masks tensor rank must be 3 or 4")
	for maskIndex in range(int(sam3Masks.shape[0])):
		mask = sam3Masks[maskIndex].detach().to(device="cpu", dtype=pt.bool).contiguous().numpy()
		result.append({"segmentation": mask})
	return result


def featureDetectionCornerOpenCVHarris(image):
	result = None
	cv2 = None
	np = None
	imageGray = None
	imageFloat32 = None
	dst = None
	cv2 = requireOpenCV()
	np = requireNumPy()
	imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	imageFloat32 = np.float32(imageGray)
	dst = cv2.cornerHarris(imageFloat32, 2, 3, 0.04)
	result = extractFeatureCoordsFromFeatureMapSubpixel(dst, imageGray)
	return result


def extractFeatureCoordsFromFeatureMapSubpixel(dst, image):
	result = None
	cv2 = None
	np = None
	thresholdValue = None
	centroids = None
	corners = None
	cv2 = requireOpenCV()
	np = requireNumPy()
	if(dst.shape[0] <= 0 or dst.shape[1] <= 0):
		raise RuntimeError("extractFeatureCoordsFromFeatureMapSubpixel error: dst shape must be > 0")
	dst = cv2.dilate(dst, None)
	thresholdValue = 0.01*float(dst.max())
	if(thresholdValue <= 0):
		result = createEmptyFeatureCoordinatesTensor()
		return result
	_, dst = cv2.threshold(dst, thresholdValue, 255, 0)
	dst = np.uint8(dst)
	_, _, _, centroids = cv2.connectedComponentsWithStats(dst)
	if(centroids.shape[0] <= 1):
		result = createEmptyFeatureCoordinatesTensor()
		return result
	corners = cv2.cornerSubPix(image, np.float32(centroids[1:]), (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))
	if(corners is None):
		result = createEmptyFeatureCoordinatesTensor()
		return result
	result = pt.tensor(corners, dtype=pt.float32, device=deviceDense)
	return result


def featureDetectionCentroidFBSegmentAnything(image):
	result = None
	centroidFeatureList = []
	segmentAnythingFeatures = None
	masks = None
	segmentationMask = None
	if(modalityORimageFeatureDetectionSegmentPostProcessing):
		segmentAnythingFeatures = detectSegmentAnythingFeatures(image)
		centroidFeatureList = convertCentroidPointsToCoordinatePairs(segmentAnythingFeatures["centroid_points"])
	else:
		masks = generateSegmentAnythingMasks(image)
		if(len(masks) > 0):
			for segmentationMask in masks:
				if("segmentation" not in segmentationMask):
					raise RuntimeError("featureDetectionCentroidFBSegmentAnything error: segmentationMask missing segmentation")
				centroidFeatureList.append(calculateMaskCentroid(segmentationMask["segmentation"]))
	if(len(centroidFeatureList) > 0):
		result = pt.tensor(centroidFeatureList, dtype=pt.float32, device=deviceDense)
	else:
		result = createEmptyFeatureCoordinatesTensor()
	return result


def printFeatureDetectionCounts(cornerFeatureCoordinates, segmentFeatureCoordinates):
	result = None
	numCornerFeatures = None
	numSegmentFeatures = None
	if(not pt.is_tensor(cornerFeatureCoordinates)):
		raise RuntimeError("printFeatureDetectionCounts error: cornerFeatureCoordinates must be a tensor")
	if(not pt.is_tensor(segmentFeatureCoordinates)):
		raise RuntimeError("printFeatureDetectionCounts error: segmentFeatureCoordinates must be a tensor")
	if(cornerFeatureCoordinates.dim() != 2):
		raise RuntimeError("printFeatureDetectionCounts error: cornerFeatureCoordinates rank must be 2")
	if(segmentFeatureCoordinates.dim() != 2):
		raise RuntimeError("printFeatureDetectionCounts error: segmentFeatureCoordinates rank must be 2")
	numCornerFeatures = int(cornerFeatureCoordinates.shape[0])
	numSegmentFeatures = int(segmentFeatureCoordinates.shape[0])
	print("featureDetection: numCornerFeaturesDetected = ", numCornerFeatures, "; numSegmentFeaturesDetected = ", numSegmentFeatures)
	return result


def printSanitisedFeatureDetectionCount(sanitisedFeatureCoordinates):
	result = None
	numSanitisedFeatureCoordinates = None
	if(not pt.is_tensor(sanitisedFeatureCoordinates)):
		raise RuntimeError("printSanitisedFeatureDetectionCount error: sanitisedFeatureCoordinates must be a tensor")
	if(sanitisedFeatureCoordinates.dim() != 2):
		raise RuntimeError("printSanitisedFeatureDetectionCount error: sanitisedFeatureCoordinates rank must be 2")
	numSanitisedFeatureCoordinates = int(sanitisedFeatureCoordinates.shape[0])
	print("featureDetection: numSanitisedUniqueFeaturesDetected = ", numSanitisedFeatureCoordinates)
	return result


def detectSegmentAnythingFeatures(image):
	result = None
	cv2 = None
	np = None
	height = None
	width = None
	masks = None
	imageContrast = None
	segmentPoints = []
	edgePoints = []
	centroidPoints = []
	colourPoints = []
	segmentationMask = None
	segmentation = None
	colourSegment = None
	segFilterPass = None
	validateRGBImage(image, "detectSegmentAnythingFeatures")
	cv2 = requireOpenCV()
	np = requireNumPy()
	height = int(image.shape[0])
	width = int(image.shape[1])
	if(height <= 0 or width <= 0):
		raise RuntimeError("detectSegmentAnythingFeatures error: image dimensions must be > 0")
	masks = generateSegmentAnythingMasks(image)
	if(len(masks) <= 0):
		result = {"segment_points": segmentPoints, "edge_points": edgePoints, "centroid_points": centroidPoints, "colour_points": colourPoints}
		return result
	imageContrast = computeContrastMap(image, "laplacian", 3)
	for segmentationMask in masks:
		segmentation = extractSegmentAnythingBinaryMask(segmentationMask)
		colourSegment = calculateSegmentMeanColour(image, segmentation)
		segFilterPass = True
		if(modalityORimageFeatureDetectionFilterSegments):
			segFilterPass = segmentPassesFilter(segmentationMask, segmentation, height, width, colourSegment)
		if(segFilterPass):
			if(modalityORimageFeatureDetectionSegmentMetadata):
				segmentPoints.append(extractSegmentPoints(segmentation))
				edgePoints.append(extractSegmentEdgePoints(segmentation, imageContrast, width, height))
				colourPoints.append(colourSegment)
			centroidPoints.append(calculateMaskCentroidWithMoments(segmentation, imageContrast))
	result = {"segment_points": segmentPoints, "edge_points": edgePoints, "centroid_points": centroidPoints, "colour_points": colourPoints}
	return result


def convertCentroidPointsToCoordinatePairs(centroidPoints):
	result = None
	coordinatePairs = []
	centroidPoint = None
	if(not isinstance(centroidPoints, list)):
		raise RuntimeError("convertCentroidPointsToCoordinatePairs error: centroidPoints must be a list")
	for centroidPoint in centroidPoints:
		if(len(centroidPoint) < 2):
			raise RuntimeError("convertCentroidPointsToCoordinatePairs error: centroidPoint must contain x and y")
		coordinatePairs.append((float(centroidPoint[0]), float(centroidPoint[1])))
	result = coordinatePairs
	return result


def calculateSegmentMeanColour(image, segmentation):
	result = None
	np = None
	mask = None
	channelIndex = None
	validateRGBImage(image, "calculateSegmentMeanColour")
	np = requireNumPy()
	if(not hasattr(segmentation, "shape")):
		raise RuntimeError("calculateSegmentMeanColour error: segmentation must be a numpy-like array")
	if(len(segmentation.shape) != 2):
		raise RuntimeError("calculateSegmentMeanColour error: segmentation rank must be 2")
	mask = segmentation.astype(bool)
	if(int(mask.sum()) <= 0):
		raise RuntimeError("calculateSegmentMeanColour error: segmentation must contain at least one foreground pixel")
	result = []
	for channelIndex in range(3):
		result.append(float(image[:, :, channelIndex][mask].mean()))
	return result


def segmentPassesFilter(segmentationMask, segmentation, imageHeight, imageWidth, colourSegment):
	result = True
	imageArea = None
	maskArea = None
	maskRatio = None
	colourSegmentLum = None
	if(not isinstance(segmentationMask, dict)):
		raise RuntimeError("segmentPassesFilter error: segmentationMask must be a dict")
	if(imageHeight <= 0 or imageWidth <= 0):
		raise RuntimeError("segmentPassesFilter error: imageHeight/imageWidth must be > 0")
	if(modalityORimageFeatureDetectionFilterSegmentsWholeImageThreshold <= 0.0 or modalityORimageFeatureDetectionFilterSegmentsWholeImageThreshold > 1.0):
		raise RuntimeError("segmentPassesFilter error: modalityORimageFeatureDetectionFilterSegmentsWholeImageThreshold must be > 0.0 and <= 1.0")
	if(modalityORimageFeatureDetectionFilterSegmentsBackgroundColourThreshold < 0.0 or modalityORimageFeatureDetectionFilterSegmentsBackgroundColourThreshold > 255.0):
		raise RuntimeError("segmentPassesFilter error: modalityORimageFeatureDetectionFilterSegmentsBackgroundColourThreshold must be >= 0.0 and <= 255.0")
	if(len(colourSegment) != 3):
		raise RuntimeError("segmentPassesFilter error: colourSegment must contain 3 channel means")
	imageArea = float(imageHeight*imageWidth)
	if("area" in segmentationMask):
		maskArea = float(segmentationMask["area"])
	else:
		maskArea = float(segmentation.sum())
	maskRatio = maskArea/imageArea
	colourSegmentLum = float(sum(colourSegment))/3.0
	if(maskRatio > modalityORimageFeatureDetectionFilterSegmentsWholeImageThreshold):
		result = False
	if(colourSegmentLum < modalityORimageFeatureDetectionFilterSegmentsBackgroundColourThreshold):
		result = False
	return result


def extractSegmentEdgePoints(segmentation, imageContrast, imageWidth, imageHeight):
	result = None
	cv2 = None
	np = None
	contourResults = None
	contours = None
	contour = None
	contourPoint = None
	edgePointsSegment = []
	ix = None
	iy = None
	contrast = None
	cv2 = requireOpenCV()
	np = requireNumPy()
	if(not hasattr(segmentation, "shape")):
		raise RuntimeError("extractSegmentEdgePoints error: segmentation must be a numpy-like array")
	if(len(segmentation.shape) != 2):
		raise RuntimeError("extractSegmentEdgePoints error: segmentation rank must be 2")
	if(not hasattr(imageContrast, "shape")):
		raise RuntimeError("extractSegmentEdgePoints error: imageContrast must be a numpy-like array")
	contourResults = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if(len(contourResults) == 2):
		contours = contourResults[0]
	elif(len(contourResults) == 3):
		contours = contourResults[1]
	else:
		raise RuntimeError("extractSegmentEdgePoints error: unexpected cv2.findContours result length")
	for contour in contours:
		for contourPoint in contour:
			ix = int(contourPoint[0][0])
			iy = int(contourPoint[0][1])
			if(ix >= 0 and ix < imageWidth and iy >= 0 and iy < imageHeight):
				contrast = float(imageContrast[iy, ix])
				edgePointsSegment.append((ix, iy, contrast))
	if(len(edgePointsSegment) > 0):
		result = np.asarray(edgePointsSegment, dtype=np.float32)
	else:
		result = np.zeros((0, 3), dtype=np.float32)
	return result


def extractSegmentPoints(segmentation):
	result = None
	np = None
	segmentPoints = None
	np = requireNumPy()
	if(not hasattr(segmentation, "shape")):
		raise RuntimeError("extractSegmentPoints error: segmentation must be a numpy-like array")
	if(len(segmentation.shape) != 2):
		raise RuntimeError("extractSegmentPoints error: segmentation rank must be 2")
	segmentPoints = np.argwhere(segmentation > 0)
	if(int(segmentPoints.shape[0]) <= 0):
		raise RuntimeError("extractSegmentPoints error: segmentation must contain at least one foreground pixel")
	result = np.asarray(segmentPoints[:, [1, 0]], dtype=np.float32)
	return result


def extractSegmentAnythingBinaryMask(segmentationMask):
	result = None
	np = None
	np = requireNumPy()
	if(not isinstance(segmentationMask, dict)):
		raise RuntimeError("extractSegmentAnythingBinaryMask error: segmentationMask must be a dict")
	if("segmentation" not in segmentationMask):
		raise RuntimeError("extractSegmentAnythingBinaryMask error: segmentationMask missing segmentation")
	result = np.asarray(segmentationMask["segmentation"], dtype=np.uint8)
	if(len(result.shape) != 2):
		raise RuntimeError("extractSegmentAnythingBinaryMask error: segmentation rank must be 2")
	result = (result > 0).astype(np.uint8)
	if(int(result.sum()) <= 0):
		raise RuntimeError("extractSegmentAnythingBinaryMask error: segmentation must contain at least one foreground pixel")
	return result


def calculateMaskCentroidWithMoments(segmentation, imageContrast):
	result = None
	cv2 = None
	moments = None
	centroidX = None
	centroidY = None
	ix = None
	iy = None
	contrast = None
	cv2 = requireOpenCV()
	if(not hasattr(segmentation, "shape")):
		raise RuntimeError("calculateMaskCentroidWithMoments error: segmentation must be a numpy-like array")
	if(len(segmentation.shape) != 2):
		raise RuntimeError("calculateMaskCentroidWithMoments error: segmentation rank must be 2")
	if(not hasattr(imageContrast, "shape")):
		raise RuntimeError("calculateMaskCentroidWithMoments error: imageContrast must be a numpy-like array")
	moments = cv2.moments(segmentation, binaryImage=True)
	if(float(moments["m00"]) == 0.0):
		raise RuntimeError("calculateMaskCentroidWithMoments error: segmentation moments m00 must be > 0")
	centroidX = float(moments["m10"]/moments["m00"])
	centroidY = float(moments["m01"]/moments["m00"])
	ix = int(round(centroidX))
	iy = int(round(centroidY))
	if(ix < 0 or ix >= int(imageContrast.shape[1]) or iy < 0 or iy >= int(imageContrast.shape[0])):
		raise RuntimeError("calculateMaskCentroidWithMoments error: centroid is outside image bounds")
	contrast = float(imageContrast[iy, ix])
	result = (centroidX, centroidY, contrast)
	return result


def calculateMaskCentroid(mask):
	result = None
	np = None
	yIndices = None
	xIndices = None
	maskedXIndices = None
	maskedYIndices = None
	centroidX = None
	centroidY = None
	np = requireNumPy()
	if(not hasattr(mask, "shape")):
		raise RuntimeError("calculateMaskCentroid error: mask must be a numpy-like array")
	yIndices, xIndices = np.indices(mask.shape)
	maskedXIndices = xIndices[mask]
	maskedYIndices = yIndices[mask]
	if(maskedXIndices.size <= 0 or maskedYIndices.size <= 0):
		raise RuntimeError("calculateMaskCentroid error: mask must contain at least one True value")
	centroidX = float(np.mean(maskedXIndices))
	centroidY = float(np.mean(maskedYIndices))
	result = (centroidX, centroidY)
	return result


def readImageRGB(imagePath):
	result = None
	cv2 = None
	imageBGR = None
	if(not isinstance(imagePath, str)):
		raise RuntimeError("readImageRGB error: imagePath must be a string")
	if(not os.path.exists(imagePath)):
		raise RuntimeError("readImageRGB error: image file not found " + str(imagePath))
	cv2 = requireOpenCV()
	imageBGR = cv2.imread(imagePath)
	if(imageBGR is None):
		raise RuntimeError("readImageRGB error: could not read image from " + str(imagePath))
	result = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
	return result


def loadImageRGB(imageSource):
	result = None
	np = None
	pillowImageModule = None
	pillowImageClass = None
	np = requireNumPy()
	if(isinstance(imageSource, str)):
		result = readImageRGB(imageSource)
	elif(pt.is_tensor(imageSource)):
		result = tensorToRGBnumpy(imageSource)
	elif(isinstance(imageSource, np.ndarray)):
		result = arrayToRGBnumpy(imageSource)
	else:
		try:
			from PIL import Image as pillowImageModuleImport
			pillowImageModule = pillowImageModuleImport
			pillowImageClass = pillowImageModule.Image
		except Exception:
			pillowImageClass = None
		if(pillowImageClass is not None and isinstance(imageSource, pillowImageClass)):
			result = np.array(imageSource.convert("RGB"))
		else:
			raise RuntimeError("loadImageRGB error: unsupported imageSource type " + str(type(imageSource)))
	validateRGBImage(result, "loadImageRGB")
	return result


def convertImageTensorToNumpyRGB(imageTensor):
	result = None
	if(not pt.is_tensor(imageTensor)):
		raise RuntimeError("convertImageTensorToNumpyRGB error: imageTensor must be a tensor")
	if(imageTensor.dim() != 3):
		raise RuntimeError("convertImageTensorToNumpyRGB error: imageTensor rank must be 3")
	if(int(imageTensor.shape[0]) != 3):
		raise RuntimeError("convertImageTensorToNumpyRGB error: imageTensor channel count must equal 3")
	result = tensorToRGBnumpy(imageTensor)
	return result


def tensorToRGBnumpy(imageTensor):
	result = None
	imageTensorCPU = None
	if(not pt.is_tensor(imageTensor)):
		raise RuntimeError("tensorToRGBnumpy error: imageTensor must be a tensor")
	if(imageTensor.dim() != 2 and imageTensor.dim() != 3):
		raise RuntimeError("tensorToRGBnumpy error: imageTensor rank must be 2 or 3")
	imageTensorCPU = imageTensor.detach().to(device="cpu", dtype=pt.float32).contiguous()
	if(imageTensorCPU.dim() == 2):
		imageTensorCPU = imageTensorCPU.unsqueeze(0)
	if(int(imageTensorCPU.shape[0]) != 1 and int(imageTensorCPU.shape[0]) != 3):
		raise RuntimeError("tensorToRGBnumpy error: imageTensor channel count must equal 1 or 3")
	if(float(imageTensorCPU.max()) <= 1.0):
		imageTensorCPU = imageTensorCPU*255.0
	imageTensorCPU = pt.clamp(imageTensorCPU, 0.0, 255.0).round().to(dtype=pt.uint8)
	result = imageTensorCPU.permute(1, 2, 0).numpy()
	if(int(result.shape[2]) == 1):
		result = result.repeat(3, axis=2)
	return result


def arrayToRGBnumpy(imageArray):
	result = None
	np = None
	np = requireNumPy()
	if(not isinstance(imageArray, np.ndarray)):
		raise RuntimeError("arrayToRGBnumpy error: imageArray must be a numpy array")
	result = imageArray
	if(result.ndim == 2):
		result = result[:, :, None]
	if(result.ndim != 3):
		raise RuntimeError("arrayToRGBnumpy error: imageArray rank must be 2 or 3")
	if(int(result.shape[2]) != 1 and int(result.shape[2]) != 3):
		if(int(result.shape[0]) == 1 or int(result.shape[0]) == 3):
			result = np.transpose(result, (1, 2, 0))
		else:
			raise RuntimeError("arrayToRGBnumpy error: imageArray must have 1 or 3 channels")
	if(int(result.shape[2]) == 1):
		result = np.repeat(result, 3, axis=2)
	if(result.dtype != np.uint8):
		result = np.asarray(result, dtype=np.float32)
		if(float(np.max(result)) <= 1.0):
			result = result*255.0
		result = np.clip(result, 0.0, 255.0).astype(np.uint8)
	return result


def validateRGBImage(image, functionName):
	result = None
	if(functionName == ""):
		raise RuntimeError("validateRGBImage error: functionName must not be empty")
	if(not hasattr(image, "shape")):
		raise RuntimeError(functionName + " error: image must be a numpy-like array")
	if(len(image.shape) != 3):
		raise RuntimeError(functionName + " error: image rank must be 3")
	if(int(image.shape[0]) <= 0 or int(image.shape[1]) <= 0):
		raise RuntimeError(functionName + " error: image dimensions must be > 0")
	if(int(image.shape[2]) != 3):
		raise RuntimeError(functionName + " error: image channel count must equal 3")
	result = image
	return result


def computeContrastMap(image, method="laplacian", ksize=3):
	result = None
	cv2 = None
	np = None
	imageGray = None
	imageContrast = None
	blurred = None
	validateRGBImage(image, "computeContrastMap")
	if(ksize <= 0 or (ksize % 2) == 0):
		raise RuntimeError("computeContrastMap error: ksize must be a positive odd integer")
	cv2 = requireOpenCV()
	np = requireNumPy()
	imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	if(method == "laplacian"):
		imageContrast = cv2.Laplacian(imageGray, cv2.CV_64F, ksize=ksize)
		imageContrast = np.abs(imageContrast)
	elif(method == "std"):
		blurred = cv2.GaussianBlur(imageGray.astype(np.float32), (ksize, ksize), 0)
		imageContrast = np.sqrt(cv2.GaussianBlur((imageGray.astype(np.float32))**2, (ksize, ksize), 0) - blurred**2)
	else:
		raise RuntimeError("computeContrastMap error: unsupported method " + str(method))
	result = cv2.normalize(imageContrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	return result


def sanitiseFeatureCoordinates(featureCoordinates, imageWidth, imageHeight):
	result = None
	validMask = None
	roundedFeatureCoordinates = None
	if(not pt.is_tensor(featureCoordinates)):
		raise RuntimeError("sanitiseFeatureCoordinates error: featureCoordinates must be a tensor")
	if(featureCoordinates.dim() != 2):
		raise RuntimeError("sanitiseFeatureCoordinates error: featureCoordinates rank must be 2")
	if(int(featureCoordinates.shape[1]) != 2):
		raise RuntimeError("sanitiseFeatureCoordinates error: featureCoordinates last dimension must equal 2")
	if(imageWidth <= 0 or imageHeight <= 0):
		raise RuntimeError("sanitiseFeatureCoordinates error: imageWidth/imageHeight must be > 0")
	if(featureCoordinates.shape[0] <= 0):
		result = createEmptyFeatureCoordinatesTensor()
		return result
	validMask = pt.isfinite(featureCoordinates).all(dim=1)
	validMask = validMask & (featureCoordinates[:, xAxisFeatureMap] >= 0.0) & (featureCoordinates[:, xAxisFeatureMap] < float(imageWidth))
	validMask = validMask & (featureCoordinates[:, yAxisFeatureMap] >= 0.0) & (featureCoordinates[:, yAxisFeatureMap] < float(imageHeight))
	result = featureCoordinates[validMask]
	if(result.shape[0] <= 0):
		result = createEmptyFeatureCoordinatesTensor()
		return result
	roundedFeatureCoordinates = pt.round(result).to(dtype=pt.int64)
	result = pt.unique(roundedFeatureCoordinates, dim=0).to(dtype=pt.float32)
	if(result.shape[0] <= 0):
		result = createEmptyFeatureCoordinatesTensor()
	return result
