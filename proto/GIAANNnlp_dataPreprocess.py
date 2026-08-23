"""GIAANNnlp_dataPreprocess.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN proto datasets

"""

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import os
from pathlib import Path
from GIAANNcmn_globalDefs import *
import TSBNLPpt_dataTokeniser

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space


def loadDataset():
	if(datasetName == 'OSCAR1900'):
		if(usePreprocessedDataset):
			if(useSmallDatasetDebug):
				dataset = load_dataset('nthngdy/oscar-small', "unshuffled_original_en", streaming=False, cache_dir=downloadCachePathName, split='train')	#or unshuffled_deduplicated_en
			else:
				dataset = load_dataset('oscar', "unshuffled_deduplicated_en", streaming=False, cache_dir=downloadCachePathName, split='train')
		else:
			dataset = load_dataset('oscar', "unshuffled_deduplicated_en", streaming=True, cache_dir=downloadCachePathName, split='train')	#https://huggingface.co/docs/datasets/stream
			#Got disconnected from remote data host. Retrying in 5sec [1/20]
		#dataset = dataset['train']
	elif(datasetName == 'OSCAR2201'):
		if(usePreprocessedDataset):
			dataset = load_dataset('oscar-corpus/OSCAR-2201', language='en', use_auth_token=tokenString, streaming=False, cache_dir=downloadCachePathName, split='train')
		else:
			dataset = load_dataset('oscar-corpus/OSCAR-2201', language='en', use_auth_token=tokenString, streaming=True, cache_dir=downloadCachePathName, split='train')	#https://huggingface.co/datasets/oscar-corpus/OSCAR-2201
	else:
		print("loadDataset error: datasetName unknown; ", datasetName)
		
	return dataset

def prepareDataElements(dataset):
	if(usePreprocessedDataset):
		dataElements = getPaths(dataPathName)
	else:
		dataElements = dataset
	return dataElements

def getPaths(dataPathName):
	if(Path(dataPathName).exists()):
		pathsGlob = Path(dataPathName).glob('**/*' + dataPreprocessedFileNameEnd)
		if(sortDataFilesByName):
			pathsGlob = sorted(pathsGlob, key=os.path.getmtime)	#key required because path names indices are not padded with 0s
		paths = [str(x) for x in pathsGlob]
	else:
		print("main error: Path does not exist, dataPathName = ", dataPathName)
		exit()
	return paths
	
def preprocessDataset(dataset):
	if(usePreprocessedDataset):
		textData = []
		fileCount = 0
		for documentIndex, document in enumerate(tqdm(dataset)):
			documentText = document['text']
			TSBNLPpt_dataTokeniser.preprocessDocumentText(documentText)
			textData.append(document)
			if(len(textData) == numberOfDocumentsPerDataFile):
				writeDataFile(fileCount, textData)
				textData = []
				fileCount += 1
			if(preprocessLimitNumberDataFiles):
				if(documentIndex == preprocessNumberOfDataFiles):
					break
		writeDataFile(fileCount, textData)	#remaining data file will be < numberOfDocumentsPerDataFile
	else:
		print("preprocessDataset error: statePreprocessDataset and !usePreprocessedDataset")
		exit()
			
def writeDataFile(fileCount, textData):
	fileName = TSBNLPpt_dataTokeniser.generateDataFileName(fileCount)
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write('\n'.join(textData))
	
def getOscar2201DocumentLengthCharacters(document):
	documentLengthCharacters = len(document['text'])	#number of characters
	'''
	meta = document['meta']
	warc_headers = meta['warc_headers']
	content_length = warc_headers['content-length']	#in bytes (characters)
	documentLengthCharacters = content_length
	'''
	return documentLengthCharacters

	
