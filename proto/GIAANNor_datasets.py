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

import os
import json
import subprocess
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNor_snapshotDimensions

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
	load_dataset = None
	DownloadConfig = None
	try:
		from datasets import load_dataset as load_datasetImport
		from datasets import DownloadConfig as DownloadConfigImport
		load_dataset = load_datasetImport
		DownloadConfig = DownloadConfigImport
	except Exception as exception:
		raise RuntimeError("requireHuggingFaceDatasets error: missing datasets (pip install datasets)") from exception
	result = load_dataset, DownloadConfig
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
	localJsonFile = downloadHuggingFaceDatasetFile(datasetJsonFile)
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


def downloadHuggingFaceDatasetFile(filename):
	result = None
	hf_hub_download = None
	datasetFolderAbsolute = None
	datasetCacheDirectory = None
	downloadCacheDirectory = None
	assetsCacheDirectory = None
	if(not isinstance(filename, str)):
		raise RuntimeError("downloadHuggingFaceDatasetFile error: filename must be a string")
	if(filename == ""):
		raise RuntimeError("downloadHuggingFaceDatasetFile error: filename must not be empty")
	_, hf_hub_download = requireHuggingFaceHub()
	if(useLocalDataset):
		datasetFolderAbsolute, datasetCacheDirectory, downloadCacheDirectory, assetsCacheDirectory = buildLocalDatasetCacheDirectories()
		result = hf_hub_download(repo_id=datasetName, repo_type="dataset", filename=filename, cache_dir=downloadCacheDirectory, local_dir=datasetFolderAbsolute, local_dir_use_symlinks=False)
	else:
		result = hf_hub_download(datasetName, repo_type="dataset", filename=filename)
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
	if("videoFile" not in datasetEntry):
		raise RuntimeError("downloadVideoFile error: datasetEntry missing videoFile")
	result = downloadHuggingFaceDatasetFile(datasetEntry["videoFile"])
	return result


def probeVideoStream(videoFile):
	result = None
	command = None
	probeProcess = None
	probeJson = None
	streamInfo = None
	if(not isinstance(videoFile, str)):
		raise RuntimeError("probeVideoStream error: videoFile must be a string")
	if(videoFile == ""):
		raise RuntimeError("probeVideoStream error: videoFile must not be empty")
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


def extractVideoSnapshots(videoFile):
	# sample the video into a series of sequence snapshots using modalityORvideoFramesPerSequenceIteration (do not apply any averaging, just skip frames as necessary).
	result = None
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
	if(not isinstance(videoFile, str)):
		raise RuntimeError("extractVideoSnapshots error: videoFile must be a string")
	if(videoFile == ""):
		raise RuntimeError("extractVideoSnapshots error: videoFile must not be empty")
	streamInfo = probeVideoStream(videoFile)
	frameWidth, frameHeight = GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(int(streamInfo["width"]), int(streamInfo["height"]), "extractVideoSnapshots")
	filterText = "select='not(mod(n\\," + str(modalityORvideoFramesPerSequenceIteration) + "))',scale=" + str(frameWidth) + ":" + str(frameHeight) + ":flags=bilinear"
	command = ["ffmpeg", "-v", "error", "-i", videoFile, "-vf", filterText, "-vsync", "vfr", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
	frameSizeBytes = frameWidth*frameHeight*3
	if(frameSizeBytes <= 0):
		raise RuntimeError("extractVideoSnapshots error: frameSizeBytes must be > 0")
	if(modalityORvideoStreamFrames):
		frameTensor = extractVideoFramesStreamFrames(command, frameWidth, frameHeight, frameSizeBytes)
	else:
		ffmpegProcess = subprocess.run(command, capture_output=True, check=True)
		rawFrameBytes = ffmpegProcess.stdout
		if(len(rawFrameBytes) % frameSizeBytes != 0):
			raise RuntimeError("extractVideoSnapshots error: ffmpeg raw frame byte count is not divisible by frameSizeBytes")
		numberOfFrames = len(rawFrameBytes)//frameSizeBytes
		if(numberOfFrames <= 0):
			raise RuntimeError("extractVideoSnapshots error: numberOfFrames must be > 0")
		frameTensor = pt.frombuffer(bytearray(rawFrameBytes), dtype=pt.uint8).clone().view(numberOfFrames, frameHeight, frameWidth, 3).permute(0, 3, 1, 2).to(deviceDense, dtype=arrayType)/255.0
	result = frameTensor
	return result


def extractVideoFramesForSnapshotSubsequences(videoFile):
	result = None
	streamInfo = None
	frameWidth = None
	frameHeight = None
	filterText = None
	command = None
	frameSizeBytes = None
	ffmpegProcess = None
	rawFrameBytes = None
	numberOfFrames = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(not isinstance(videoFile, str)):
			raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: videoFile must be a string")
		if(videoFile == ""):
			raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: videoFile must not be empty")
		if(not isinstance(modalityORvideoFrameRate, int) and not isinstance(modalityORvideoFrameRate, float)):
			raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: modalityORvideoFrameRate must be an int or float")
		if(modalityORvideoFrameRate <= 0):
			raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: modalityORvideoFrameRate must be > 0")
		if(not isinstance(modalityORvideoFramesPerSequenceIteration, int)):
			raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: modalityORvideoFramesPerSequenceIteration must be an int")
		if(modalityORvideoFramesPerSequenceIteration <= 0):
			raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: modalityORvideoFramesPerSequenceIteration must be > 0")
		streamInfo = probeVideoStream(videoFile)
		frameWidth = int(streamInfo["width"])
		frameHeight = int(streamInfo["height"])
		GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(frameWidth, frameHeight, "extractVideoFramesForSnapshotSubsequences")
		filterText = "fps=" + str(modalityORvideoFrameRate) + ",select='not(mod(n\\," + str(modalityORvideoFramesPerSequenceIteration) + "))'"
		command = ["ffmpeg", "-v", "error", "-i", videoFile, "-vf", filterText, "-vsync", "vfr", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
		frameSizeBytes = frameWidth*frameHeight*3
		if(frameSizeBytes <= 0):
			raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: frameSizeBytes must be > 0")
		if(modalityORvideoStreamFrames):
			result = extractVideoFramesStreamFrames(command, frameWidth, frameHeight, frameSizeBytes)
		else:
			ffmpegProcess = subprocess.run(command, capture_output=True, check=True)
			rawFrameBytes = ffmpegProcess.stdout
			if(len(rawFrameBytes) % frameSizeBytes != 0):
				raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: ffmpeg raw frame byte count is not divisible by frameSizeBytes")
			numberOfFrames = len(rawFrameBytes)//frameSizeBytes
			if(numberOfFrames <= 0):
				raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: numberOfFrames must be > 0")
			result = pt.frombuffer(bytearray(rawFrameBytes), dtype=pt.uint8).clone().view(numberOfFrames, frameHeight, frameWidth, 3).permute(0, 3, 1, 2).to(deviceDense, dtype=arrayType)/255.0
	else:
		raise RuntimeError("extractVideoFramesForSnapshotSubsequences error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def extractVideoFramesStreamFrames(command, frameWidth, frameHeight, frameSizeBytes):
	result = None
	ffmpegProcess = None
	frameTensors = None
	rawFrameBytes = None
	stderrBytes = None
	returnCode = None
	numberOfFrames = None
	if(not isinstance(command, list)):
		raise RuntimeError("extractVideoFramesStreamFrames error: command must be a list")
	if(frameWidth <= 0 or frameHeight <= 0 or frameSizeBytes <= 0):
		raise RuntimeError("extractVideoFramesStreamFrames error: frameWidth/frameHeight/frameSizeBytes must be > 0")
	frameTensors = []
	ffmpegProcess = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if(ffmpegProcess.stdout is None or ffmpegProcess.stderr is None):
		raise RuntimeError("extractVideoFramesStreamFrames error: failed to open ffmpeg pipes")
	while True:
		rawFrameBytes = ffmpegProcess.stdout.read(frameSizeBytes)
		if(rawFrameBytes == b""):
			break
		if(len(rawFrameBytes) != frameSizeBytes):
			ffmpegProcess.kill()
			ffmpegProcess.wait()
			raise RuntimeError("extractVideoFramesStreamFrames error: incomplete raw frame read from ffmpeg stdout")
		frameTensors.append(pt.frombuffer(bytearray(rawFrameBytes), dtype=pt.uint8).clone().view(frameHeight, frameWidth, 3).permute(2, 0, 1))
	stderrBytes = ffmpegProcess.stderr.read()
	returnCode = ffmpegProcess.wait()
	if(returnCode != 0):
		raise RuntimeError("extractVideoFramesStreamFrames error: ffmpeg failed with stderr=" + stderrBytes.decode("utf-8", errors="replace"))
	numberOfFrames = len(frameTensors)
	if(numberOfFrames <= 0):
		raise RuntimeError("extractVideoFramesStreamFrames error: numberOfFrames must be > 0")
	result = pt.stack(frameTensors, dim=0).to(deviceDense, dtype=arrayType)/255.0
	return result


def loadImageDatasetSplit(split):
	result = None
	load_dataset = None
	DownloadConfig = None
	datasetFolderAbsolute = None
	datasetCacheDirectory = None
	downloadCacheDirectory = None
	assetsCacheDirectory = None
	downloadConfig = None
	streaming = None
	if(datasetType != "cifar10" and datasetType != "cityscapes"):
		raise RuntimeError("loadImageDatasetSplit error: unsupported image datasetType " + str(datasetType))
	if(split != "train" and split != "test"):
		raise RuntimeError("loadImageDatasetSplit error: split must be 'train' or 'test'")
	if(useLocalDataset and datasetFolder == ""):
		raise RuntimeError("loadImageDatasetSplit error: datasetFolder must not be empty while useLocalDataset is True")
	if(useLocalDataset):
		datasetFolderAbsolute, datasetCacheDirectory, downloadCacheDirectory, assetsCacheDirectory = buildLocalDatasetCacheDirectories()
	streaming = not useLocalDataset
	if(split not in _imageDatasetCache):
		load_dataset, DownloadConfig = requireHuggingFaceDatasets()
		if(useLocalDataset):
			downloadConfig = DownloadConfig(cache_dir=downloadCacheDirectory)
		if(datasetCfg == ""):
			if(useLocalDataset):
				_imageDatasetCache[split] = load_dataset(datasetName, split=split, streaming=streaming, trust_remote_code=True, cache_dir=datasetCacheDirectory, download_config=downloadConfig)
			else:
				_imageDatasetCache[split] = load_dataset(datasetName, split=split, streaming=streaming, trust_remote_code=True)
		else:
			if(useLocalDataset):
				_imageDatasetCache[split] = load_dataset(datasetName, datasetCfg, split=split, streaming=streaming, trust_remote_code=True, cache_dir=datasetCacheDirectory, download_config=downloadConfig)
			else:
				_imageDatasetCache[split] = load_dataset(datasetName, datasetCfg, split=split, streaming=streaming, trust_remote_code=True)
	result = _imageDatasetCache[split]
	return result


def buildLocalDatasetCacheDirectories():
	result = None
	datasetFolderAbsolute = None
	datasetCacheDirectory = None
	downloadCacheDirectory = None
	assetsCacheDirectory = None
	if(datasetFolder == ""):
		raise RuntimeError("buildLocalDatasetCacheDirectories error: datasetFolder must not be empty")
	if(os.path.exists(datasetFolder) and not os.path.isdir(datasetFolder)):
		raise RuntimeError("buildLocalDatasetCacheDirectories error: datasetFolder exists but is not a directory: " + str(datasetFolder))
	os.makedirs(datasetFolder, exist_ok=True)
	datasetFolderAbsolute = os.path.abspath(datasetFolder)
	datasetCacheDirectory = os.path.join(datasetFolderAbsolute, "datasets_cache")
	downloadCacheDirectory = os.path.join(datasetFolderAbsolute, "downloads_cache")
	assetsCacheDirectory = os.path.join(datasetFolderAbsolute, "assets_cache")
	os.makedirs(datasetCacheDirectory, exist_ok=True)
	os.makedirs(downloadCacheDirectory, exist_ok=True)
	os.makedirs(assetsCacheDirectory, exist_ok=True)
	os.environ["HF_HOME"] = datasetFolderAbsolute
	os.environ["HF_DATASETS_CACHE"] = datasetCacheDirectory
	os.environ["HF_HUB_CACHE"] = downloadCacheDirectory
	os.environ["HUGGINGFACE_HUB_CACHE"] = downloadCacheDirectory
	os.environ["HF_ASSETS_CACHE"] = assetsCacheDirectory
	result = datasetFolderAbsolute, datasetCacheDirectory, downloadCacheDirectory, assetsCacheDirectory
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
