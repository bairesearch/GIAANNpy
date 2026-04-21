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

import json
import subprocess
import torch as pt

from GIAANNcmn_globalDefs import *

_datasetIndexCache = None


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
	if(datasetType != "soccer_events"):
		raise RuntimeError("loadDataset error: unsupported OR datasetType " + str(datasetType))
	if(split != "train" and split != "test"):
		raise RuntimeError("loadDataset error: split must be 'train' or 'test'")
	datasetEntries = loadDatasetIndex()
	trainEntries, testEntries = splitDatasetEntries(datasetEntries)
	if(split == "train"):
		result = trainEntries
	else:
		result = testEntries
	return result


def loadPromptDataset():
	result = None
	testEntries = loadDataset("test")
	if(len(testEntries) > modalityORdatasetPromptMaxSequences):
		result = testEntries[:modalityORdatasetPromptMaxSequences]
	else:
		result = testEntries
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
	# sample the video into a series of sequence snapshots using modalityORframesPerSnapshot (do not apply any averaging, just skip frames as necessary).
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
	filterText = "select='not(mod(n\\," + str(modalityORframesPerSnapshot) + "))',scale=" + str(frameWidth) + ":" + str(frameHeight) + ":flags=bilinear"
	command = ["ffmpeg", "-v", "error", "-i", videoFile, "-vf", filterText, "-vsync", "vfr", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
	ffmpegProcess = subprocess.run(command, capture_output=True, check=True)
	rawFrameBytes = ffmpegProcess.stdout
	frameSizeBytes = frameWidth*frameHeight*3
	if(frameSizeBytes <= 0):
		raise RuntimeError("sampleVideoSnapshots error: frameSizeBytes must be > 0")
	if(len(rawFrameBytes) % frameSizeBytes != 0):
		raise RuntimeError("sampleVideoSnapshots error: ffmpeg raw frame byte count is not divisible by frameSizeBytes")
	numberOfFrames = len(rawFrameBytes)//frameSizeBytes
	if(numberOfFrames <= 0):
		raise RuntimeError("sampleVideoSnapshots error: numberOfFrames must be > 0")
	frameTensor = pt.frombuffer(bytearray(rawFrameBytes), dtype=pt.uint8).clone().view(numberOfFrames, frameHeight, frameWidth, 3).permute(0, 3, 1, 2).to(deviceDense, dtype=arrayType)/255.0
	result = frameTensor
	return result
