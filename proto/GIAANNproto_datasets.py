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

def loadWikipediaDataset():
	dataset = None
	if(useLocalDataset):
		if(datasetFolder == ""):
			raise RuntimeError("loadWikipediaDataset error: datasetFolder is empty while useLocalDataset is True")
		if(not os.path.isdir(datasetFolder)):
			os.makedirs(datasetFolder, exist_ok=True)
		if(useLocalDatasetDownloadManual):
			dataset = loadLocalDatasetManual()
		else:
			dataset = load_dataset(datasetName, datasetCfg, split="train", streaming=False, trust_remote_code=True, cache_dir=datasetFolder)
	else:
		dataset = load_dataset(datasetName, datasetCfg, split="train", streaming=True, trust_remote_code=True)
	return dataset

if(useLocalDatasetDownloadManual):

	def loadLocalDatasetManual():
		dataset = None
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
		if(not datasetsLibrary4plus):
			repoId = "legacy-datasets/wikipedia"
		return repoId

	def getDatasetFilePrefix():
		filePrefix = "data/" + datasetCfg + "/"
		if(datasetsLibrary4plus):
			filePrefix = datasetCfg + "/"
		return filePrefix
