"""GIAANNcmn_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:

## common:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
python -m pip install --upgrade pip
pip install networkx
pip install matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
datasetsLibrary4plus=False: pip install "datasets<4" "fsspec==2024.6.1" "gcsfs==2024.6.1"

## modality NLP:
pip install spacy
python -m spacy download en_core_web_sm [spacyModelName]
pip install nltk

## modality OR:
pip install opencv-python
pip install git+https://github.com/facebookresearch/segment-anything.git [if modalityORfeatureDetectionSAMversion==1]
pip install git+https://github.com/facebookresearch/sam2.git [if modalityORfeatureDetectionSAMversion==2]
pip install git+https://github.com/facebookresearch/sam3.git [if modalityORfeatureDetectionSAMversion==3]
hf auth login [if modalityORfeatureDetectionSAMversion==3 and modalityORfeatureDetectionSAM3checkpoint==""]

# Usage:
source activate pytorchsenv
python GIAANNcmn_main.py

# Description:
GIA ANN common main

"""

# Import necessary libraries
import gc
import time
import torch as pt
import spacy

pt.set_grad_enabled(False)

from GIAANNcmn_globalDefs import *
import GIAANNcmn_debug
import GIAANNcmn_count
import GIAANNcmn_databaseNetwork
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_databaseNetworkDrawLarge

if(modalityName=="NLP"):
	import GIAANNnlp_main
elif(modalityName=="OR"):
	import GIAANNor_main
		
if(executionMode=="inference" or executionMode=="trainAndInference"):
	import GIAANNcmn_prediction
if(modalityName=="NLP" and inferenceReportGroundedAccuracy):
	import GIAANNnlp_groundedEval


if(printTimeDatabaseLoadSaveTimes):
	countPrintTimeDatabaseLoadSaveTimesExecutionModeCount = GIAANNcmn_count.getCountPrintTimeDatabaseLoadSaveTimesExecutionModeCount()
	countPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount = 0
	countPrintTimeDatabaseLoadSaveTimesProgramExecutionStartTime = 0.0
	countPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime = 0.0
	countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime = 0.0
	countPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesExecuteModeStartTime = 0.0
	countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime = 0.0

if(printTimeDatabaseLoadSaveTimes):
	countPrintTimeDatabaseLoadSaveTimesProgramExecutionStartTime = time.perf_counter()

if(useAutoresearch):
	autoresearchExecutionTimeTrain = None
	autoresearchExecutionTimeInference = None

# Load the selected dataset using Hugging Face datasets
if(datasetType != "textfile" and executionMode != "inference" and not useDrawNetworkIndependently):
	if(modalityName=="NLP"):
		import GIAANNnlp_datasets
	elif(modalityName=="OR"):
		import GIAANNor_datasets
	if(printTimeDatabaseLoadSaveTimes):
		debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadStartTime = time.perf_counter()
	if(modalityName=="NLP"):
		dataset = GIAANNnlp_datasets.loadDataset()
	elif(modalityName=="OR"):
		dataset = GIAANNor_datasets.loadDataset()
	if(printTimeDatabaseLoadSaveTimes):
		countPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime = countPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime + (time.perf_counter() - debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadStartTime)

if(debugPrintSpacySectionTimes):
	processArticlePart1totalTime = 0
	processArticlePart2totalTime = 0
	processArticlePart1count = 0
	processArticlePart2count = 0

def main():
	GIAANNcmn_databaseNetworkFiles.prepareDatabaseFilesStartup()
	
	if(printRamMaxUsage or useAutoresearch):
		GIAANNcmn_count.countResetGpuRamMaxUsage()
	if(debugPrintRamMaxUsagePhaseLocal):
		GIAANNcmn_debug.debugResetPhaseLocalProgramPeakTracking()
	
	if(useDrawNetworkIndependently):
		executeDrawMode()
	else:
		if(executionMode=="inference"):
			executeMode(True)
		elif(executionMode=="trainAndInference"):
			executeMode(False) 
			executeMode(True)
		elif(executionMode=="train"):
			executeMode(False)
	
	if(printRamMaxUsage):
		GIAANNcmn_count.countPrintGpuRamMaxUsageSummary()
	if(debugPrintRamAverageUsage and not debugPrintRamCurrentUsage):
		GIAANNcmn_debug.debugPrintRamUsageSummary()
	if(debugPrintRamMaxUsagePhaseLocal):
		GIAANNcmn_debug.debugPrintGpuRamMaxUsagePhaseLocalSummary()
		GIAANNcmn_debug.debugPrintPhaseLocalProgramPeakSummary()


def getDrawModeDatabaseInferenceMode():
	if(executionMode == "train"):
		inferenceMode = False
	elif(executionMode == "trainAndInference"):
		inferenceMode = True
	elif(executionMode == "inference"):
		inferenceMode = True
	else:
		raise RuntimeError(f"getDrawModeDatabaseInferenceMode error: unsupported executionMode {executionMode}")
	return inferenceMode

def validateDrawModeExistingDatabaseFiles():
	if(not GIAANNcmn_databaseNetworkFiles.pathExists(conceptColumnsDictFile)):
		raise RuntimeError(f"validateDrawModeExistingDatabaseFiles error: missing conceptColumnsDictFile {conceptColumnsDictFile}")
	if(not GIAANNcmn_databaseNetworkFiles.pathExists(conceptFeaturesDictFile)):
		raise RuntimeError(f"validateDrawModeExistingDatabaseFiles error: missing conceptFeaturesDictFile {conceptFeaturesDictFile}")
	if(storeDatabaseGlobalFeatureNeuronsInRam):
		if(not GIAANNcmn_databaseNetworkFiles.pathExists(globalFeatureNeuronsFileFull)):
			raise RuntimeError(f"validateDrawModeExistingDatabaseFiles error: missing globalFeatureNeuronsFileFull {globalFeatureNeuronsFileFull}")
	return

def executeDrawMode():
	validateDrawModeExistingDatabaseFiles()
	databaseNetworkObject = GIAANNcmn_databaseNetwork.initialiseDatabaseNetwork(getDrawModeDatabaseInferenceMode(), loadExistingDatabaseOverride=True)
	if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
		GIAANNcmn_databaseNetwork.loadAllObservedColumnsToRam(databaseNetworkObject)
	GIAANNcmn_databaseNetworkDrawLarge.drawDatabaseGraphStandalone(databaseNetworkObject, save=True, fileName=drawNetworkIndependentSaveFilename, display=False)
	return
	
def executeMode(inferenceMode):
	if(printTimeDatabaseLoadSaveTimes):
		global debugPrintTimeDatabaseLoadSaveTimesExecuteModeStartTime
		global countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime
		global countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime
		global debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime
		global countPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime
		global countPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount
		debugPrintTimeDatabaseLoadSaveTimesExecuteModeStartTime = time.perf_counter()
	if(useAutoresearch):
		autoresearchExecutionTimeStart = time.perf_counter()

	databaseNetworkObject = GIAANNcmn_databaseNetwork.initialiseDatabaseNetwork(inferenceMode)
	if(modalityName=="NLP"):
		databaseNetworkObject.nlp = GIAANNnlp_main.nlpSequence	#used by posStringToPosInt
	if(inferenceMode):
		if(inferenceStartGenerateGlobalFeatureNeuronsTensor):
			GIAANNcmn_databaseNetwork.generateHighMemGlobalFeatureNeuronsForInferenceStartup(databaseNetworkObject)

	if(printCountTotalParameters):
		if(len(databaseNetworkObject.conceptColumnsList) > 0):
			GIAANNcmn_count.printCountTotalParametersRun(databaseNetworkObject)
		else:
			print("printCountTotalParameters totalColumns = 0 (empty database)")
	
	if(modalityName=="NLP"):
		GIAANNnlp_main.loadPOSdatabase()
	if(inferenceMode and not inferenceTrainFirstSequences):
		GIAANNcmn_databaseNetwork.backupGlobalArrays(databaseNetworkObject)
	
	if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
		if(printTimeDatabaseLoadSaveTimes):
			countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime = time.perf_counter()
		GIAANNcmn_databaseNetwork.loadAllObservedColumnsToRam(databaseNetworkObject)
		if(printTimeDatabaseLoadSaveTimes):
			countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime = countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime + (time.perf_counter() - countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime)
		
	for epochIndex in range(numberEpochs):
		#print("\nepochIndex = ", epochIndex)
		# Start processing the dataset
		sequenceCount = 0
		if(inferenceMode and debugPrintTotalInferenceTokens):
			GIAANNcmn_debug.resetTotalInferenceTokens()
		if(inferenceMode):
			GIAANNcmn_prediction.resetInferenceTop1AccuracyCounts()
			if(modalityName=="NLP" and inferenceReportGroundedAccuracy):
				GIAANNnlp_groundedEval.resetInferenceGroundedAccuracyCounts()
		
		if(modalityName=="NLP"):
			if(inferenceMode):
				sequenceCount = GIAANNnlp_main.processPrompt(databaseNetworkObject, inferenceMode, sequenceCount)
			else:
				sequenceCount = GIAANNnlp_main.processDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset)
		elif(modalityName=="OR"):
			if(inferenceMode):
				sequenceCount = GIAANNor_main.processPrompt(databaseNetworkObject, inferenceMode, sequenceCount)
			else:
				sequenceCount = GIAANNor_main.processDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset)

		if(inferenceMode and debugPrintTotalInferenceTokens):
			GIAANNcmn_debug.printTotalInferenceTokens()
		if(inferenceMode and printInferenceTop1Accuracy and not useAutoresearch):
			GIAANNcmn_prediction.printInferenceTop1Accuracy(databaseNetworkObject)
		if(inferenceMode and inferenceReportGroundedAccuracy and not useAutoresearch):
			GIAANNnlp_groundedEval.printInferenceGroundedAccuracy(databaseNetworkObject)

	if(debugPrintSpacySectionTimes):
		processArticlePart1averageTime = processArticlePart1totalTime/processArticlePart1count
		processArticlePart2averageTime = processArticlePart2totalTime/processArticlePart2count
		print(f"debugPrintSpacySectionTimes: processArticlePart1averageTime={processArticlePart1averageTime:.6f} processArticlePart2averageTime={processArticlePart2averageTime:.6f}")

	if(not inferenceMode or inferenceTrainFirstSequences):
		if(useSaveData):
			if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("saveAllObservedColumnsToDisk")
				if(printTimeDatabaseLoadSaveTimes):
					debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime = time.perf_counter()
				GIAANNcmn_databaseNetwork.saveAllObservedColumnsToDisk(databaseNetworkObject)
				if(printTimeDatabaseLoadSaveTimes):
					countPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime = countPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime + (time.perf_counter() - debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime)
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("saveAllObservedColumnsToDisk")
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("saveData(final)")
			GIAANNcmn_databaseNetworkFiles.saveData(databaseNetworkObject, {}, sequenceCount, forceSaveGlobalState=True)
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("saveData(final)")
			#only required if trainMaxSequences%saveGlobalFeatureNeuronsRate != 0

	if(printTimeDatabaseLoadSaveTimes):
		countPrintTimeDatabaseLoadSaveTimesProgramExecutionEndTime = time.perf_counter()
		countPrintTimeDatabaseLoadSaveTimesTotalExecutionTime = countPrintTimeDatabaseLoadSaveTimesProgramExecutionEndTime - countPrintTimeDatabaseLoadSaveTimesProgramExecutionStartTime
		countPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount = countPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount + 1
		if(countPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount == countPrintTimeDatabaseLoadSaveTimesExecutionModeCount):
			countPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime = countPrintTimeDatabaseLoadSaveTimesTotalExecutionTime - countPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime
			if(countPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime < 0):
				raise RuntimeError("executeMode error: countPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime must be >= 0")
			GIAANNcmn_count.printCountPrintTimeDatabaseLoadSaveTimesEntry("Hugging Face dataset load execution time", countPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime)
			GIAANNcmn_count.printCountPrintTimeDatabaseLoadSaveTimesSummary("printTimeDatabaseLoadSaveTimes execution times:", countPrintTimeDatabaseLoadSaveTimesTotalExecutionTime, countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime, countPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime)
			GIAANNcmn_count.printCountPrintTimeDatabaseLoadSaveTimesSummary("printTimeDatabaseLoadSaveTimes execution times with Hugging Face dataset load time subtracted from total execution time:", countPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime, countPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime, countPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime)
	if(useAutoresearch):
		autoresearchExecutionTimeEnd = time.perf_counter()
		autoresearchExecutionTime = autoresearchExecutionTimeEnd - autoresearchExecutionTimeStart
		if(inferenceMode):
			global autoresearchExecutionTimeInference
			autoresearchExecutionTimeInference = autoresearchExecutionTime
		else:
			global autoresearchExecutionTimeTrain
			autoresearchExecutionTimeTrain = autoresearchExecutionTime
		if(inferenceMode and printInferenceTop1Accuracy):
			GIAANNcmn_prediction.printInferenceTop1Accuracy(databaseNetworkObject, autoresearchExecutionTimeInference, autoresearchExecutionTimeTrain)
		if(inferenceMode and inferenceReportGroundedAccuracy):
			GIAANNnlp_groundedEval.printInferenceGroundedAccuracy(databaseNetworkObject, autoresearchExecutionTimeInference, autoresearchExecutionTimeTrain)
			

if __name__ == "__main__":
	with pt.no_grad():
		main()
