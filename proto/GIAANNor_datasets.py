"""GIAANNor_datasets.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN or datasets

"""

import math
import json
import subprocess
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNor_keypoints

_datasetIndexCache = None
_imageDatasetCache = {}


def requireHuggingFaceHub():
	HfApi = None
	hf_hub_download = None
	try:
		from huggingface_hub import HfApi as HfApiImport, hf_hub_download as hf_hub_download_import
		HfApi = HfApiImport
		hf_hub_download = hf_hub_download_import
	except Exception as exception:
		raise RuntimeError("requireHuggingFaceHub error: missing huggingface_hub (pip install huggingface_hub)") from exception
	return HfApi, hf_hub_download


def requireHuggingFaceDatasets():
	result = None
	try:
		from datasets import load_dataset as load_datasetImport
		result = load_datasetImport
	except Exception as exception:
		raise RuntimeError("requireHuggingFaceDatasets error: missing datasets (pip install datasets)") from exception
	return result


def timecodeToSeconds(timecode, frameRate):
	result = None
	if(not isinstance(timecode, str)):
		raise RuntimeError("timecodeToSeconds error: timecode must be a str")
	if(not isinstance(frameRate, float) and not isinstance(frameRate, int)):
		raise RuntimeError("timecodeToSeconds error: frameRate must be a float or int")
	if(frameRate <= 0):
		raise RuntimeError("timecodeToSeconds error: frameRate must be > 0")
	timecodeParts = timecode.split(":")
	if(len(timecodeParts) != 4):
		raise RuntimeError("timecodeToSeconds error: timecode must contain 4 components")
	hours = int(timecodeParts[0])
	minutes = int(timecodeParts[1])
	seconds = int(timecodeParts[2])
	frames = int(timecodeParts[3])
	result = float(hours*3600 + minutes*60 + seconds) + (float(frames)/float(frameRate))
	return result


def buildDatasetEntry(datasetJsonFile):
	result = None
	frameRate = None
	startSeconds = None
	endSeconds = None
	durationSeconds = None
	if(not datasetJsonFile.startswith("data/") or not datasetJsonFile.endswith(".json")):
		raise RuntimeError("buildDatasetEntry error: datasetJsonFile must be a data/*.json file")
	_, hf_hub_download = requireHuggingFaceHub()
	localJsonFile = hf_hub_download(datasetName, repo_type="dataset", filename=datasetJsonFile)
	with open(localJsonFile, "r", encoding="utf-8") as fileObject:
		metadata = json.load(fileObject)
	if("asset_id" not in metadata):
		raise RuntimeError("buildDatasetEntry error: metadata missing asset_id")
	if("frame_rate" not in metadata):
		raise RuntimeError("buildDatasetEntry error: metadata missing frame_rate")
	if("starttimecode" not in metadata or "endtimecode" not in metadata):
		raise RuntimeError("buildDatasetEntry error: metadata missing starttimecode/endtimecode")
	frameRate = float(metadata["frame_rate"])
	startSeconds = timecodeToSeconds(metadata["starttimecode"], frameRate)
	endSeconds = timecodeToSeconds(metadata["endtimecode"], frameRate)
	durationSeconds = endSeconds - startSeconds
	if(durationSeconds <= 0):
		raise RuntimeError("buildDatasetEntry error: durationSeconds must be > 0")
	result = {"assetId": metadata["asset_id"], "jsonFile": datasetJsonFile, "videoFile": datasetJsonFile.replace(".json", ".mp4"), "metadata": metadata, "frameRate": frameRate, "durationSeconds": durationSeconds}
	return result


def isDatasetEntryEligible(datasetEntry):
	result = False
	durationSeconds = float(datasetEntry["durationSeconds"])
	if(durationSeconds >= modalityORvideoMinDurationSeconds and durationSeconds <= modalityORvideoMaxDurationSeconds):
		result = True
	return result


def loadDatasetIndex():
	# add huggingface datasetType for video (find an appropriate dataset for short approx 1-3 minute lengthed, low to moderate resolution, video streams).
	global _datasetIndexCache
	result = None
	if(_datasetIndexCache is None):
		HfApi, _ = requireHuggingFaceHub()
		api = HfApi()
		repoFiles = api.list_repo_files(datasetName, repo_type="dataset")
		jsonFiles = []
		for repoFile in repoFiles:
			if(repoFile.startswith("data/") and repoFile.endswith(".json")):
				jsonFiles.append(repoFile)
		jsonFiles.sort()
		if(len(jsonFiles) == 0):
			raise RuntimeError("loadDatasetIndex error: no dataset json files found")
		eligibleEntries = []
		for datasetJsonFile in jsonFiles:
			datasetEntry = buildDatasetEntry(datasetJsonFile)
			if(isDatasetEntryEligible(datasetEntry)):
				eligibleEntries.append(datasetEntry)
		if(len(eligibleEntries) == 0):
			raise RuntimeError("loadDatasetIndex error: no eligible dataset entries found after duration filtering")
		_datasetIndexCache = eligibleEntries
	result = list(_datasetIndexCache)
	return result


def splitDatasetEntries(datasetEntries):
	trainEntries = []
	testEntries = []
	promptCount = 0
	if(not isinstance(datasetEntries, list)):
		raise RuntimeError("splitDatasetEntries error: datasetEntries must be a list")
	if(len(datasetEntries) < 2):
		raise RuntimeError("splitDatasetEntries error: datasetEntries must contain at least 2 entries")
	promptCount = int(round(float(len(datasetEntries))*float(modalityORdatasetPromptRatio)))
	if(promptCount <= 0):
		promptCount = 1
	if(promptCount >= len(datasetEntries)):
		raise RuntimeError("splitDatasetEntries error: promptCount must be < len(datasetEntries)")
	trainEntries = datasetEntries[:-promptCount]
	testEntries = datasetEntries[-promptCount:]
	if(len(trainEntries) == 0 or len(testEntries) == 0):
		raise RuntimeError("splitDatasetEntries error: trainEntries/testEntries must not be empty")
	return trainEntries, testEntries


def loadDataset(split="train"):
	result = None
	datasetEntries = None
	trainEntries = None
	testEntries = None
	if(split != "train" and split != "test"):
		raise RuntimeError("loadDataset error: split must be 'train' or 'test'")
	if(datasetType == "soccer_events"):
		datasetEntries = loadDatasetIndex()
		trainEntries, testEntries = splitDatasetEntries(datasetEntries)
		if(split == "train"):
			result = trainEntries
		else:
			result = testEntries
	elif(datasetType == "cifar10" or datasetType == "cityscapes"):
		result = loadImageDatasetSplit(split)
	else:
		raise RuntimeError("loadDataset error: unsupported OR datasetType " + str(datasetType))
	return result


def loadPromptDataset():
	result = None
	testEntries = loadDataset("test")
	result = limitDatasetEntries(testEntries, modalityORdatasetPromptMaxSequences)
	return result


def limitDatasetEntries(datasetEntries, maxEntries):
	result = None
	datasetLength = None
	if(not isinstance(maxEntries, int)):
		raise RuntimeError("limitDatasetEntries error: maxEntries must be an int")
	if(maxEntries <= 0):
		raise RuntimeError("limitDatasetEntries error: maxEntries must be > 0")
	if(hasattr(datasetEntries, "take")):
		result = datasetEntries.take(maxEntries)
	elif(hasattr(datasetEntries, "select")):
		datasetLength = len(datasetEntries)
		if(datasetLength > maxEntries):
			result = datasetEntries.select(range(maxEntries))
		else:
			result = datasetEntries
	else:
		datasetLength = len(datasetEntries)
		if(datasetLength > maxEntries):
			result = datasetEntries[:maxEntries]
		else:
			result = datasetEntries
	return result


def downloadVideoFile(datasetEntry):
	result = None
	_, hf_hub_download = requireHuggingFaceHub()
	result = hf_hub_download(datasetName, repo_type="dataset", filename=datasetEntry["videoFile"])
	return result


def probeVideoStream(videoFile):
	result = None
	command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", videoFile]
	probeProcess = subprocess.run(command, capture_output=True, text=True, check=True)
	probeJson = json.loads(probeProcess.stdout)
	if("streams" not in probeJson or len(probeJson["streams"]) != 1):
		raise RuntimeError("probeVideoStream error: ffprobe returned an unexpected stream description")
	streamInfo = probeJson["streams"][0]
	if("width" not in streamInfo or "height" not in streamInfo):
		raise RuntimeError("probeVideoStream error: ffprobe missing width/height")
	result = {"width": int(streamInfo["width"]), "height": int(streamInfo["height"])}
	return result


def sampleVideoSnapshots(datasetEntry):
	# sample the video into a series of sequence snapshots using modalityORvideoFramesPerSnapshot (do not apply any averaging, just skip frames as necessary).
	result = None
	videoFile = None
	streamInfo = None
	filterText = None
	command = None
	ffmpegProcess = None
	rawFrameBytes = None
	frameWidth = None
	frameHeight = None
	frameSizeBytes = None
	numberOfFrames = None
	frameTensor = None
	videoFile = downloadVideoFile(datasetEntry)
	streamInfo = probeVideoStream(videoFile)
	frameWidth = int(modalityORsnapshotWidth)
	frameHeight = int(modalityORsnapshotHeight)
	if(frameWidth <= 0 or frameHeight <= 0):
		raise RuntimeError("sampleVideoSnapshots error: modalityORsnapshotWidth/modalityORsnapshotHeight must be > 0")
	filterText = "select='not(mod(n\\," + str(modalityORvideoFramesPerSnapshot) + "))',scale=" + str(frameWidth) + ":" + str(frameHeight) + ":flags=bilinear"
	command = ["ffmpeg", "-v", "error", "-i", videoFile, "-vf", filterText, "-vsync", "vfr", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
	frameSizeBytes = frameWidth*frameHeight*3
	if(frameSizeBytes <= 0):
		raise RuntimeError("sampleVideoSnapshots error: frameSizeBytes must be > 0")
	if(modalityORvideoStreamFrames):
		frameTensor = sampleVideoSnapshotsStreamFrames(command, frameWidth, frameHeight, frameSizeBytes)
	else:
		ffmpegProcess = subprocess.run(command, capture_output=True, check=True)
		rawFrameBytes = ffmpegProcess.stdout
		if(len(rawFrameBytes) % frameSizeBytes != 0):
			raise RuntimeError("sampleVideoSnapshots error: ffmpeg raw frame byte count is not divisible by frameSizeBytes")
		numberOfFrames = len(rawFrameBytes)//frameSizeBytes
		if(numberOfFrames <= 0):
			raise RuntimeError("sampleVideoSnapshots error: numberOfFrames must be > 0")
		frameTensor = pt.frombuffer(bytearray(rawFrameBytes), dtype=pt.uint8).clone().view(numberOfFrames, frameHeight, frameWidth, 3).permute(0, 3, 1, 2).to(deviceDense, dtype=arrayType)/255.0
	result = frameTensor
	return result


def sampleVideoSnapshotsStreamFrames(command, frameWidth, frameHeight, frameSizeBytes):
	result = None
	ffmpegProcess = None
	frameTensors = None
	rawFrameBytes = None
	stderrBytes = None
	returnCode = None
	numberOfFrames = None
	if(not isinstance(command, list)):
		raise RuntimeError("sampleVideoSnapshotsStreamFrames error: command must be a list")
	if(frameWidth <= 0 or frameHeight <= 0 or frameSizeBytes <= 0):
		raise RuntimeError("sampleVideoSnapshotsStreamFrames error: frameWidth/frameHeight/frameSizeBytes must be > 0")
	frameTensors = []
	ffmpegProcess = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if(ffmpegProcess.stdout is None or ffmpegProcess.stderr is None):
		raise RuntimeError("sampleVideoSnapshotsStreamFrames error: failed to open ffmpeg pipes")
	while True:
		rawFrameBytes = ffmpegProcess.stdout.read(frameSizeBytes)
		if(rawFrameBytes == b""):
			break
		if(len(rawFrameBytes) != frameSizeBytes):
			ffmpegProcess.kill()
			ffmpegProcess.wait()
			raise RuntimeError("sampleVideoSnapshotsStreamFrames error: incomplete raw frame read from ffmpeg stdout")
		frameTensors.append(pt.frombuffer(bytearray(rawFrameBytes), dtype=pt.uint8).clone().view(frameHeight, frameWidth, 3).permute(2, 0, 1))
	stderrBytes = ffmpegProcess.stderr.read()
	returnCode = ffmpegProcess.wait()
	if(returnCode != 0):
		raise RuntimeError("sampleVideoSnapshotsStreamFrames error: ffmpeg failed with stderr=" + stderrBytes.decode("utf-8", errors="replace"))
	numberOfFrames = len(frameTensors)
	if(numberOfFrames <= 0):
		raise RuntimeError("sampleVideoSnapshotsStreamFrames error: numberOfFrames must be > 0")
	result = pt.stack(frameTensors, dim=0).to(deviceDense, dtype=arrayType)/255.0
	return result


def loadImageDatasetSplit(split):
	result = None
	load_dataset = None
	streaming = None
	if(datasetType != "cifar10" and datasetType != "cityscapes"):
		raise RuntimeError("loadImageDatasetSplit error: unsupported image datasetType " + str(datasetType))
	if(split != "train" and split != "test"):
		raise RuntimeError("loadImageDatasetSplit error: split must be 'train' or 'test'")
	if(useLocalDataset and datasetFolder == ""):
		raise RuntimeError("loadImageDatasetSplit error: datasetFolder must not be empty while useLocalDataset is True")
	streaming = not useLocalDataset
	if(split not in _imageDatasetCache):
		load_dataset = requireHuggingFaceDatasets()
		if(datasetCfg == ""):
			if(useLocalDataset):
				_imageDatasetCache[split] = load_dataset(datasetName, split=split, streaming=streaming, trust_remote_code=True, cache_dir=datasetFolder)
			else:
				_imageDatasetCache[split] = load_dataset(datasetName, split=split, streaming=streaming, trust_remote_code=True)
		else:
			if(useLocalDataset):
				_imageDatasetCache[split] = load_dataset(datasetName, datasetCfg, split=split, streaming=streaming, trust_remote_code=True, cache_dir=datasetFolder)
			else:
				_imageDatasetCache[split] = load_dataset(datasetName, datasetCfg, split=split, streaming=streaming, trust_remote_code=True)
	result = _imageDatasetCache[split]
	return result


def sampleImageSaccadeSequences(datasetEntry):
	# generate sequence data by augmenting each image:
	# for each image, generate modalityORimageSaccadesPerImage sequences by performing modalityORimageSaccadesPerImage augmentations:
	result = []
	preparedImageTensor = None
	cropMarginX = None
	cropMarginY = None
	preparedImageWorkWidth = None
	preparedImageWorkHeight = None
	saccadeOffsetPairs = None
	saccadeOffsetPair = None
	targetOffsetX = None
	targetOffsetY = None
	sequence = None
	if(modalityORimageSaccadesPerImage <= 0):
		raise RuntimeError("sampleImageSaccadeSequences error: modalityORimageSaccadesPerImage must be > 0")
	if(modalityORimageSnapshotsPerSaccade <= 0):
		raise RuntimeError("sampleImageSaccadeSequences error: modalityORimageSnapshotsPerSaccade must be > 0")
	if(modalityORimageSaccadesCrop):
		cropMarginX, cropMarginY = calculateImageSaccadeCropMargins()
		preparedImageWorkWidth = int(modalityORsnapshotWidth) + (2*cropMarginX)
		preparedImageWorkHeight = int(modalityORsnapshotHeight) + (2*cropMarginY)
		preparedImageTensor = prepareImageTensorForSaccades(datasetEntry, preparedImageWorkWidth, preparedImageWorkHeight)
	else:
		preparedImageTensor = prepareImageTensorForSaccades(datasetEntry, int(modalityORsnapshotWidth), int(modalityORsnapshotHeight))
		cropMarginX, cropMarginY = calculateImageSaccadeCropMarginsFromPreparedImage(preparedImageTensor)
	if(submodalityName=="image"):
		if(modalityORimageSaccadesUseAdjacentSalientRegions):
			saccadeOffsetPairs = GIAANNor_keypoints.sampleAdjacentSalientImageSaccadeOffsetPairs(preparedImageTensor, cropMarginX, cropMarginY)
			if(saccadeOffsetPairs is None):
				result = None
			else:
				for saccadeOffsetPair in saccadeOffsetPairs:
					sequence = generateImageSaccadeSequenceBetweenOffsets(preparedImageTensor, float(saccadeOffsetPair[0].item()), float(saccadeOffsetPair[1].item()), float(saccadeOffsetPair[2].item()), float(saccadeOffsetPair[3].item()), cropMarginX, cropMarginY)
					result.append(sequence)
		else:
			for _ in range(modalityORimageSaccadesPerImage):
				targetOffsetX, targetOffsetY = sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY)
				sequence = generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY)
				result.append(sequence)
	else:
		for _ in range(modalityORimageSaccadesPerImage):
			targetOffsetX, targetOffsetY = sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY)
			sequence = generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY)
			result.append(sequence)
	return result


def generateImageSaccadeSequenceBetweenOffsets(preparedImageTensor, startOffsetX, startOffsetY, endOffsetX, endOffsetY, cropMarginX, cropMarginY):
	result = None
	snapshotSequenceTensor = None
	workHeight = None
	workWidth = None
	snapshotIndexFraction = None
	snapshotOffsetX = None
	snapshotOffsetY = None
	startX = None
	startY = None
	endX = None
	endY = None
	snapshotTensor = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: preparedImageTensor rank must be 3")
	if(preparedImageTensor.shape[0] != 3):
		raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: preparedImageTensor channel count must be 3")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	snapshotSequenceTensor = pt.zeros((modalityORimageSnapshotsPerSaccade, 3, modalityORsnapshotHeight, modalityORsnapshotWidth), dtype=arrayType, device=deviceDense)
	for snapshotIndex in range(modalityORimageSnapshotsPerSaccade):
		if(modalityORimageSnapshotsPerSaccade == 1):
			snapshotIndexFraction = 1.0
		else:
			snapshotIndexFraction = float(snapshotIndex)/float(modalityORimageSnapshotsPerSaccade - 1)
		snapshotOffsetX = float(startOffsetX) + (float(endOffsetX - startOffsetX)*snapshotIndexFraction)
		snapshotOffsetY = float(startOffsetY) + (float(endOffsetY - startOffsetY)*snapshotIndexFraction)
		startX = int(cropMarginX) + int(round(snapshotOffsetX))
		startY = int(cropMarginY) + int(round(snapshotOffsetY))
		endX = startX + int(modalityORsnapshotWidth)
		endY = startY + int(modalityORsnapshotHeight)
		if(startX < 0 or startY < 0 or endX > workWidth or endY > workHeight):
			raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: computed crop window exceeds preparedImageTensor bounds")
		snapshotTensor = preparedImageTensor[:, startY:endY, startX:endX]
		if(snapshotTensor.shape[1] != modalityORsnapshotHeight or snapshotTensor.shape[2] != modalityORsnapshotWidth):
			raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: snapshotTensor shape mismatch")
		snapshotSequenceTensor[snapshotIndex] = snapshotTensor
	result = snapshotSequenceTensor
	return result


def generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY):
	# for each saccade (sequence) generate modalityORimageSnapshotsPerSaccade by taking snapshots along a linear pathway of the saccade offset:
	# crop each augmented snapshot by a predefined amount (dependent on modalityORimageSaccadesMaxAngularOffsetDegrees) so that every snapshot contains only image data (no pixels outside the original image data).
	result = None
	snapshotSequenceTensor = None
	workHeight = None
	workWidth = None
	snapshotIndexFraction = None
	snapshotOffsetX = None
	snapshotOffsetY = None
	startX = None
	startY = None
	endX = None
	endY = None
	snapshotTensor = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("generateImageSaccadeSequence error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("generateImageSaccadeSequence error: preparedImageTensor rank must be 3")
	if(preparedImageTensor.shape[0] != 3):
		raise RuntimeError("generateImageSaccadeSequence error: preparedImageTensor channel count must be 3")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	snapshotSequenceTensor = pt.zeros((modalityORimageSnapshotsPerSaccade, 3, modalityORsnapshotHeight, modalityORsnapshotWidth), dtype=arrayType, device=deviceDense)
	for snapshotIndex in range(modalityORimageSnapshotsPerSaccade):
		if(modalityORimageSnapshotsPerSaccade == 1):
			snapshotIndexFraction = 1.0
		else:
			snapshotIndexFraction = float(snapshotIndex)/float(modalityORimageSnapshotsPerSaccade - 1)
		snapshotOffsetX = int(round(float(targetOffsetX)*snapshotIndexFraction))
		snapshotOffsetY = int(round(float(targetOffsetY)*snapshotIndexFraction))
		startX = int(cropMarginX) + snapshotOffsetX
		startY = int(cropMarginY) + snapshotOffsetY
		endX = startX + int(modalityORsnapshotWidth)
		endY = startY + int(modalityORsnapshotHeight)
		if(startX < 0 or startY < 0 or endX > workWidth or endY > workHeight):
			raise RuntimeError("generateImageSaccadeSequence error: computed crop window exceeds preparedImageTensor bounds")
		snapshotTensor = preparedImageTensor[:, startY:endY, startX:endX]
		if(snapshotTensor.shape[1] != modalityORsnapshotHeight or snapshotTensor.shape[2] != modalityORsnapshotWidth):
			raise RuntimeError("generateImageSaccadeSequence error: snapshotTensor shape mismatch")
		snapshotSequenceTensor[snapshotIndex] = snapshotTensor
	result = snapshotSequenceTensor
	return result


def sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY):
	# saccade augmentations are calculated by translating the image to a random polar coordinates offset from the centre (using modalityORimageSaccadesMaxAngularOffsetDegrees)
	result = None
	angleRadians = None
	radiusScale = None
	offsetX = None
	offsetY = None
	if(cropMarginX < 0 or cropMarginY < 0):
		raise RuntimeError("sampleRandomImageSaccadeOffset error: cropMarginX/cropMarginY must be >= 0")
	angleRadians = float(pt.rand(1).item())*(2.0*math.pi)
	radiusScale = float(pt.rand(1).item())
	offsetX = math.cos(angleRadians)*float(cropMarginX)*radiusScale
	offsetY = math.sin(angleRadians)*float(cropMarginY)*radiusScale
	result = (offsetX, offsetY)
	return result


def calculateImageSaccadeCropMargins():
	result = None
	angleRadians = None
	cropMarginX = None
	cropMarginY = None
	if(modalityORsnapshotWidth <= 0 or modalityORsnapshotHeight <= 0):
		raise RuntimeError("calculateImageSaccadeCropMargins error: modalityORsnapshotWidth/modalityORsnapshotHeight must be > 0")
	if(modalityORimageSaccadesMaxAngularOffsetDegrees < 0 or modalityORimageSaccadesMaxAngularOffsetDegrees >= 90):
		raise RuntimeError("calculateImageSaccadeCropMargins error: modalityORimageSaccadesMaxAngularOffsetDegrees must be >= 0 and < 90")
	angleRadians = math.radians(float(modalityORimageSaccadesMaxAngularOffsetDegrees))
	cropMarginX = int(math.ceil((float(modalityORsnapshotWidth)/2.0)*math.tan(angleRadians)))
	cropMarginY = int(math.ceil((float(modalityORsnapshotHeight)/2.0)*math.tan(angleRadians)))
	result = (cropMarginX, cropMarginY)
	return result


def calculateImageSaccadeCropMarginsFromPreparedImage(preparedImageTensor):
	result = None
	workHeight = None
	workWidth = None
	cropMarginX = None
	cropMarginY = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: preparedImageTensor rank must be 3")
	if(preparedImageTensor.shape[0] != 3):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: preparedImageTensor channel count must be 3")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	if(workWidth < int(modalityORsnapshotWidth) or workHeight < int(modalityORsnapshotHeight)):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: prepared image must be at least as large as modalityORsnapshotWidth/modalityORsnapshotHeight")
	cropMarginX = int((float(workWidth) - float(modalityORsnapshotWidth))/2.0)
	cropMarginY = int((float(workHeight) - float(modalityORsnapshotHeight))/2.0)
	result = (cropMarginX, cropMarginY)
	return result


def prepareImageTensorForSaccades(datasetEntry, workWidth, workHeight):
	result = None
	imageTensor = None
	imageHeight = None
	imageWidth = None
	cropStartX = None
	cropStartY = None
	if(workWidth <= 0 or workHeight <= 0):
		raise RuntimeError("prepareImageTensorForSaccades error: workWidth/workHeight must be > 0")
	imageTensor = convertDatasetEntryToImageTensor(datasetEntry)
	imageHeight = int(imageTensor.shape[1])
	imageWidth = int(imageTensor.shape[2])
	if(imageWidth < workWidth or imageHeight < workHeight):
		raise RuntimeError("prepareImageTensorForSaccades error: input image must cover workWidth/workHeight without resizing")
	if(modalityORimageSaccadesCrop):
		cropStartX = int((imageWidth - workWidth)//2)
		cropStartY = int((imageHeight - workHeight)//2)
		result = imageTensor[:, cropStartY:cropStartY + workHeight, cropStartX:cropStartX + workWidth].contiguous()
		if(result.shape[1] != workHeight or result.shape[2] != workWidth):
			raise RuntimeError("prepareImageTensorForSaccades error: prepared image crop shape mismatch")
	else:
		result = imageTensor.contiguous()
		if(result.shape[1] < workHeight or result.shape[2] < workWidth):
			raise RuntimeError("prepareImageTensorForSaccades error: prepared image shape must cover workWidth/workHeight")
	return result


def convertDatasetEntryToImageTensor(datasetEntry):
	result = None
	imageObject = None
	imageWidth = None
	imageHeight = None
	rawImageBytes = None
	expectedByteCount = None
	if(not isinstance(datasetEntry, dict)):
		raise RuntimeError("convertDatasetEntryToImageTensor error: datasetEntry must be a dict")
	if("img" in datasetEntry):
		imageObject = datasetEntry["img"]
	elif("image" in datasetEntry):
		imageObject = datasetEntry["image"]
	else:
		raise RuntimeError("convertDatasetEntryToImageTensor error: datasetEntry missing img/image field")
	if(not hasattr(imageObject, "convert") or not hasattr(imageObject, "tobytes") or not hasattr(imageObject, "size")):
		raise RuntimeError("convertDatasetEntryToImageTensor error: imageObject must provide convert/tobytes/size")
	imageObject = imageObject.convert("RGB")
	imageWidth, imageHeight = imageObject.size
	if(imageWidth <= 0 or imageHeight <= 0):
		raise RuntimeError("convertDatasetEntryToImageTensor error: image size must be > 0")
	rawImageBytes = imageObject.tobytes()
	expectedByteCount = int(imageWidth)*int(imageHeight)*3
	if(len(rawImageBytes) != expectedByteCount):
		raise RuntimeError("convertDatasetEntryToImageTensor error: rawImageBytes length mismatch")
	result = pt.frombuffer(bytearray(rawImageBytes), dtype=pt.uint8).clone().view(imageHeight, imageWidth, 3).permute(2, 0, 1).contiguous().to(deviceDense, dtype=arrayType)/255.0
	return result
