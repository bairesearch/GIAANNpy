"""Small deterministic fixtures for the benchmark runner tests."""
benchmarkTestSequenceDelay = 0.4
benchmarkTestPoll = 0.05
benchmarkTestDeadline = 30
benchmarkTestTokensPerSequence = 10
benchmarkTestSecondMatches = 7
benchmarkTestFirstMatches = 3
benchmarkTestConfigurationText = "proto/GIAANNcmn_globalDefs.py:line1:matchesPerSequence = 7\nproto/extra.py:line1:value = 'changed:content'\n"
benchmarkTestTorch = "__version__ = 'test-double'\ndef set_num_threads(count):\n\treturn\ndef set_num_interop_threads(count):\n\treturn\n"
benchmarkTestPrediction = 'totalInferenceTop1PredictionMatches = 0\ntotalInferenceTop1PredictionTokens = 0\n'
benchmarkTestMain = "import time\nfrom pathlib import Path\nimport GIAANNcmn_globalDefs as defs\nimport GIAANNnlp_main as nlp\ndef main():\n\tfor index, line in enumerate(Path(defs.inferencePromptFile).read_text().splitlines()):\n\t\tif(line.strip()):\n\t\t\tnlp.processSequence(None, True, index, line)\n\treturn\n"
benchmarkTestNLP = "import time\nfrom pathlib import Path\nimport GIAANNcmn_globalDefs as defs\nimport GIAANNcmn_prediction as prediction\ndef processSequence(database, inference, index, text):\n\tif(text == 'second' and Path(defs.failureMarker).exists()):\n\t\traise RuntimeError('deliberate fixture failure')\n\tif(Path(defs.writeMarker).exists()):\n\t\t(Path(defs.databaseFolder) / 'must-not-write.txt').write_text('bad')\n\ttime.sleep(defs.testDelay)\n\tprediction.totalInferenceTop1PredictionMatches += defs.matchesPerSequence\n\tprediction.totalInferenceTop1PredictionTokens += defs.testTokens\n\treturn\n"
