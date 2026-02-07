"""GIAANNproto_datasets.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto datasets

"""


import os
import shutil
from datasets import load_dataset, load_from_disk

from GIAANNproto_globalDefs import *

def loadDatasetFromHuggingFace(streaming, cacheDirectory):
	dataset = None
	if(not isinstance(streaming, bool)):
		raise RuntimeError("loadDatasetFromHuggingFace error: streaming must be a bool")
	if(cacheDirectory is not None and not isinstance(cacheDirectory, str)):
		raise RuntimeError("loadDatasetFromHuggingFace error: cacheDirectory must be a string or None")
	if(datasetCfg == ""):
		if(cacheDirectory is None):
			dataset = load_dataset(datasetName, split="train", streaming=streaming, trust_remote_code=True)
		else:
			dataset = load_dataset(datasetName, split="train", streaming=streaming, trust_remote_code=True, cache_dir=cacheDirectory)
	else:
		if(cacheDirectory is None):
			dataset = load_dataset(datasetName, datasetCfg, split="train", streaming=streaming, trust_remote_code=True)
		else:
			dataset = load_dataset(datasetName, datasetCfg, split="train", streaming=streaming, trust_remote_code=True, cache_dir=cacheDirectory)
	return dataset

def loadDataset():
	dataset = None
	if(useLocalDataset):
		if(datasetFolder == ""):
			raise RuntimeError("loadDataset error: datasetFolder is empty while useLocalDataset is True")
		if(not os.path.isdir(datasetFolder)):
			os.makedirs(datasetFolder, exist_ok=True)
		if(useLocalDatasetDownloadManual):
			dataset = loadLocalDatasetManual()
		else:
			if(datasetWikipedia):
				dataset = loadDatasetFromHuggingFace(False, datasetFolder)
			elif(datasetOscar):
				dataset = loadDatasetFromHuggingFace(True, datasetFolder)
			else:
				raise RuntimeError("loadDataset error: unsupported dataset selection for non-manual local loading")
	else:
		dataset = loadDatasetFromHuggingFace(True, None)
	if(trainTestSet):
		dataset = applyTrainTestSetOffset(dataset)
	return dataset

def loadWikipediaDataset():
	dataset = loadDataset()
	return dataset

def getDatasetEntryText(datasetEntry, articleIndex):
	text = None
	if(not isinstance(articleIndex, int)):
		raise RuntimeError("getDatasetEntryText error: articleIndex must be an int")
	if(articleIndex < 0):
		raise RuntimeError("getDatasetEntryText error: articleIndex must be >= 0")
	if(isinstance(datasetEntry, str)):
		text = datasetEntry
	elif(isinstance(datasetEntry, dict)):
		if("text" in datasetEntry and isinstance(datasetEntry["text"], str)):
			text = datasetEntry["text"]
		elif("content" in datasetEntry and isinstance(datasetEntry["content"], str)):
			text = datasetEntry["content"]
		elif("article" in datasetEntry and isinstance(datasetEntry["article"], str)):
			text = datasetEntry["article"]
		elif("document" in datasetEntry and isinstance(datasetEntry["document"], str)):
			text = datasetEntry["document"]
		elif("body" in datasetEntry and isinstance(datasetEntry["body"], str)):
			text = datasetEntry["body"]
		else:
			raise RuntimeError("getDatasetEntryText error: unsupported dataset entry fields at articleIndex=" + str(articleIndex) + ", keys=" + str(list(datasetEntry.keys())))
	else:
		raise RuntimeError("getDatasetEntryText error: unsupported dataset entry type at articleIndex=" + str(articleIndex) + ", type=" + str(type(datasetEntry)))
	return text

def applyTrainTestSetOffset(dataset):
	updatedDataset = dataset
	if(trainTestSet):
		if(datasetWikipedia):
			if(not isinstance(testSetRatio, float) and not isinstance(testSetRatio, int)):
				raise RuntimeError("applyTrainTestSetOffset error: testSetRatio is not a float or int")
			if(testSetRatio <= 0 or testSetRatio >= 1):
				raise RuntimeError("applyTrainTestSetOffset error: testSetRatio must be > 0 and < 1 when trainTestSet is True")
			try:
				datasetLength = len(updatedDataset)
			except Exception as e:
				raise RuntimeError("applyTrainTestSetOffset error: failed to read dataset length for dynamic trainTestSetArticleOffset") from e
			if(datasetLength <= 0):
				raise RuntimeError("applyTrainTestSetOffset error: dataset length <= 0")
			trainTestSetArticleOffset = int(float(datasetLength) * (1.0 - float(testSetRatio)))
			if(trainTestSetArticleOffset <= 0 or trainTestSetArticleOffset >= datasetLength):
				raise RuntimeError("applyTrainTestSetOffset error: computed trainTestSetArticleOffset is out of range (" + str(trainTestSetArticleOffset) + " of " + str(datasetLength) + ")")
			if(hasattr(updatedDataset, "skip")):
				updatedDataset = updatedDataset.skip(trainTestSetArticleOffset)
			elif(hasattr(updatedDataset, "select")):
				updatedDataset = updatedDataset.select(range(trainTestSetArticleOffset, datasetLength))
			else:
				raise RuntimeError("applyTrainTestSetOffset error: dataset does not support skip or select")
		elif(datasetOscar):
			if(not isinstance(testSetStartOffset, int)):
				raise RuntimeError("applyTrainTestSetOffset error: testSetStartOffset is not an int")
			if(not isinstance(testSetSize, int)):
				raise RuntimeError("applyTrainTestSetOffset error: testSetSize is not an int")
			if(testSetStartOffset < 0):
				raise RuntimeError("applyTrainTestSetOffset error: testSetStartOffset must be >= 0 when trainTestSet is True")
			if(testSetSize <= 0):
				raise RuntimeError("applyTrainTestSetOffset error: testSetSize must be > 0 when trainTestSet is True")
			if(hasattr(updatedDataset, "skip") and hasattr(updatedDataset, "take")):
				updatedDataset = updatedDataset.skip(testSetStartOffset)
				updatedDataset = updatedDataset.take(testSetSize)
				datasetHasEntries = False
				datasetIterator = iter(updatedDataset)
				try:
					next(datasetIterator)
					datasetHasEntries = True
				except StopIteration:
					datasetHasEntries = False
				if(not datasetHasEntries):
					raise RuntimeError("applyTrainTestSetOffset error: selected test set is empty; reduce testSetStartOffset or increase dataset size")
			elif(hasattr(updatedDataset, "select")):
				try:
					datasetLength = len(updatedDataset)
				except Exception as e:
					raise RuntimeError("applyTrainTestSetOffset error: failed to read dataset length for static test-set offset") from e
				if(datasetLength <= 0):
					raise RuntimeError("applyTrainTestSetOffset error: dataset length <= 0")
				if(testSetStartOffset >= datasetLength):
					raise RuntimeError("applyTrainTestSetOffset error: testSetStartOffset out of range (" + str(testSetStartOffset) + " of " + str(datasetLength) + ")")
				testSetEndOffset = testSetStartOffset + testSetSize
				if(testSetEndOffset > datasetLength):
					raise RuntimeError("applyTrainTestSetOffset error: testSetStartOffset + testSetSize exceeds dataset length (" + str(testSetEndOffset) + " of " + str(datasetLength) + ")")
				updatedDataset = updatedDataset.select(range(testSetStartOffset, testSetEndOffset))
			else:
				raise RuntimeError("applyTrainTestSetOffset error: dataset does not support skip+take or select")
		else:
			raise RuntimeError("applyTrainTestSetOffset error: unsupported dataset selection for trainTestSet")
	return updatedDataset

if(useLocalDatasetDownloadManual):

	def loadLocalDatasetManual():
		dataset = None
		if(not datasetWikipedia):
			raise RuntimeError("loadLocalDatasetManual error: manual dataset download currently only supports datasetWikipedia=True")
		if(datasetProcessedCacheFolder == ""):
			raise RuntimeError("loadLocalDatasetManual error: datasetProcessedCacheFolder is empty while useLocalDatasetDownloadManual is True")
		if(os.path.exists(datasetProcessedCacheFolder) and not os.path.isdir(datasetProcessedCacheFolder)):
			raise RuntimeError("loadLocalDatasetManual error: datasetProcessedCacheFolder is not a directory: " + datasetProcessedCacheFolder)
		processedDatasetInfoFile = datasetProcessedCacheFolder + "dataset_info.json"
		if(os.path.isdir(datasetProcessedCacheFolder) and os.path.isfile(processedDatasetInfoFile)):
			print("loadLocalDatasetManual: loading processed dataset cache from " + datasetProcessedCacheFolder)
			dataset = load_from_disk(datasetProcessedCacheFolder)
		elif(os.path.isdir(datasetProcessedCacheFolder) and not os.path.isfile(processedDatasetInfoFile)):
			raise RuntimeError("loadLocalDatasetManual error: datasetProcessedCacheFolder exists but missing dataset_info.json: " + datasetProcessedCacheFolder)
		else:
			print("loadLocalDatasetManual: building processed dataset cache at " + datasetProcessedCacheFolder)
			os.makedirs(datasetProcessedCacheFolder, exist_ok=True)
			hfCacheFolder = datasetProcessedCacheFolder + "hf_cache/"
			os.makedirs(hfCacheFolder, exist_ok=True)
			datasetFiles = downloadLocalDatasetFiles()
			localFiles = buildLocalDatasetFilePaths(datasetFiles)
			dataset = load_dataset("parquet", data_files=localFiles, split="train", cache_dir=hfCacheFolder)
			dataset.save_to_disk(datasetProcessedCacheFolder)
			shutil.rmtree(hfCacheFolder)
			if(os.path.exists(hfCacheFolder)):
				raise RuntimeError("loadLocalDatasetManual error: failed to remove hfCacheFolder: " + hfCacheFolder)
		return dataset

	def getManualDatasetFileList():
		datasetFiles = []
		HfApi, hf_hub_download = requireHuggingFaceHub()
		api = HfApi()
		repoId = getDatasetRepoId()
		repoFiles = api.list_repo_files(repo_id=repoId, repo_type="dataset")
		filePrefix = getDatasetFilePrefix()
		for filePath in repoFiles:
			if(filePath.startswith(filePrefix) and filePath.endswith(".parquet")):
				datasetFiles.append(filePath)
		if(len(datasetFiles) == 0):
			raise RuntimeError("getManualDatasetFileList error: no parquet files found for datasetCfg=" + datasetCfg + " in repoId=" + repoId)
		datasetFiles.sort()
		return datasetFiles

	def downloadLocalDatasetFiles():
		datasetFiles = []
		HfApi, hf_hub_download = requireHuggingFaceHub()
		repoId = getDatasetRepoId()
		datasetFiles = getManualDatasetFileList()
		for filePath in datasetFiles:
			localFilePath = os.path.join(datasetFolder, filePath)
			if(not os.path.isfile(localFilePath)):
				hf_hub_download(repo_id=repoId, repo_type="dataset", filename=filePath, local_dir=datasetFolder, local_dir_use_symlinks=False)
		return datasetFiles

	def buildLocalDatasetFilePaths(datasetFiles):
		localFiles = []
		for filePath in datasetFiles:
			localFilePath = os.path.join(datasetFolder, filePath)
			if(not os.path.isfile(localFilePath)):
				raise RuntimeError("buildLocalDatasetFilePaths error: missing local file " + localFilePath)
			localFiles.append(localFilePath)
		return localFiles

	def requireHuggingFaceHub():
		HfApi = None
		hf_hub_download = None
		try:
			from huggingface_hub import HfApi as HfApiImport, hf_hub_download as hf_hub_download_import
			HfApi = HfApiImport
			hf_hub_download = hf_hub_download_import
		except Exception as e:
			raise RuntimeError("requireHuggingFaceHub error: missing huggingface_hub (pip install huggingface_hub)") from e
		return HfApi, hf_hub_download

	def getDatasetRepoId():
		repoId = datasetName
		if(datasetWikipedia and not datasetsLibrary4plus):
			repoId = "legacy-datasets/wikipedia"
		elif(not datasetWikipedia):
			raise RuntimeError("getDatasetRepoId error: manual dataset download currently only supports datasetWikipedia=True")
		return repoId

	def getDatasetFilePrefix():
		filePrefix = "data/" + datasetCfg + "/"
		if(datasetWikipedia and datasetsLibrary4plus):
			filePrefix = datasetCfg + "/"
		elif(not datasetWikipedia):
			raise RuntimeError("getDatasetFilePrefix error: manual dataset download currently only supports datasetWikipedia=True")
		return filePrefix
