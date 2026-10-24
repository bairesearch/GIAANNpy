"""Deterministic model fixtures; tests execute the real proto chunk scheduler and worker."""
benchmarkTestSequenceDelay = 0.2
benchmarkTestPoll = 0.05
benchmarkTestDeadline = 40
benchmarkTestTokensPerSequence = 10
benchmarkTestSecondMatches = 7
benchmarkTestFirstMatches = 3
benchmarkTestSequenceCount = 3
benchmarkTestConfigurationCount = 2
benchmarkTestWorkerCount = 2
benchmarkTestChunkSize = 2
benchmarkTestSeedTokens = 1
benchmarkTestPredictionModule = 'GIAANNcmn_prediction'
benchmarkTestNLPModule = 'GIAANNnlp_main'
benchmarkTestDebugModule = 'GIAANNcmn_debug'
benchmarkTestIgnoreFilename = '.gitignore'
benchmarkTestHistoricalDirectory = 'historical'
benchmarkTestHistoricalFilename = 'invalid.py'
benchmarkTestInvalidSource = 'if broken syntax\n'
benchmarkTestRepeatCount = 2
benchmarkTestInput = 'first text\nshort\n\nsecond text\nthird text\n'
benchmarkTestStalePrompt = 'stale database prompt\n'
benchmarkTestConfigurationText = "proto/GIAANNcmn_globalDefs.py:line1:matchesPerSequence = 7\nproto/extra.py:line1:value = 'changed:content'\n"
benchmarkTestTorch = "from contextlib import nullcontext\n__version__ = 'fixture'\ndef set_num_threads(count):\n\treturn\ndef set_num_interop_threads(count):\n\treturn\ndef no_grad():\n\treturn nullcontext()\n"
benchmarkTestPrediction = 'totalInferenceTop1Matches = 0\ntotalInferenceTop1Tokens = 0\ntotalInferenceTop1PredictionMatches = 0\ntotalInferenceTop1PredictionTokens = 0\ndef resetInferenceTop1AccuracyCounts():\n\tglobal totalInferenceTop1Matches,totalInferenceTop1Tokens,totalInferenceTop1PredictionMatches,totalInferenceTop1PredictionTokens\n\ttotalInferenceTop1Matches = totalInferenceTop1Tokens = totalInferenceTop1PredictionMatches = totalInferenceTop1PredictionTokens = 0\n\treturn\ndef printInferenceTop1Accuracy(database):\n\treturn\n'
benchmarkTestDebug = 'totalInferenceTokensSeed = 0\ntotalInferenceTokensPrediction = 0\ntotalInferenceTokensAll = 0\n'
benchmarkTestMain = 'from pathlib import Path\nimport GIAANNcmn_globalDefs as defs\nimport GIAANNcmn_inferenceChunks as chunks\nimport GIAANNnlp_main as nlp\ndef main():\n\tchunks.executeInferenceChunks()\n\treturn\ndef mainStandard():\n\tlines = [s for s in Path(defs.inferencePromptFile).read_text().splitlines() if s.strip() and len(s.split()) > defs.numSeedTokensInference]\n\tfor index,line in enumerate(lines):\n\t\tnlp.processSequence(None,True,index,0,index,line)\n\treturn\n'
benchmarkTestNLP = "import time\nfrom pathlib import Path\nimport GIAANNcmn_globalDefs as defs\nimport GIAANNcmn_prediction as prediction\nclass Parser:\n\tdef pipe(self, lines):\n\t\treturn [line.split() for line in lines]\nnlpSequence = Parser()\ndef processSequence(database,inference,count,article,index,text):\n\tif(text.startswith('second') and defs.matchesPerSequence==defs.failingMatches and Path(defs.failureMarker).exists()):\n\t\traise RuntimeError('deliberate fixture failure')\n\tif(Path(defs.writeMarker).exists()):\n\t\t(Path(defs.databaseFolder)/'must-not-write.txt').write_text('bad')\n\ttime.sleep(defs.testDelay)\n\tprediction.totalInferenceTop1PredictionMatches += defs.matchesPerSequence\n\tprediction.totalInferenceTop1PredictionTokens += defs.testTokens\n\tprediction.totalInferenceTop1Matches += defs.matchesPerSequence+defs.numSeedTokensInference\n\tprediction.totalInferenceTop1Tokens += defs.testTokens+defs.numSeedTokensInference\n\treturn\n"
benchmarkTestNativeRequired = {'executionMode': 'inference', 'modalityName': 'NLP', 'useTrainDuringInference': False, 'inferenceTrainFirstSequences': False, 'inferenceStartGenerateGlobalFeatureNeuronsTensor': False, 'trainLoadExistingDatabase': True, 'useSaveData': True, 'inferencePromptPreserveLineSequenceBoundaries': True, 'numberEpochs': 1, 'trainSetStartOffsetSequences': 0, 'useGPUdense': False, 'useGPUsparse': False, 'useGPUfileio': False, 'storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam': False, 'useGPUdatabase': False, 'useAutoresearch': False, 'useDrawNetworkIndependently': False, 'drawNetworkDuringInference': False, 'drawNetworkDuringInferenceSave': False, 'inferenceReportGroundedAccuracy': False, 'datasetType': 'fixture', 'closedWorldGroundedDatasetTypes': [], 'printInferenceTop1AccuracyBitsPerByte': False, 'printInferenceTop1Accuracy': True, 'debugPrintTotalInferenceTokens': False, 'inferencePromptExpectedSequencesPerLine': 1, 'inferenceCopyTemplateDatasets': True, 'debugDisableDatabaseFileWrite': False, 'optimisationInferenceChunks': True, 'optimisationInferenceWorkers': True, 'inferenceChunksWorkerThreads': 1, 'inferenceChunksOutputDirectory': None}
