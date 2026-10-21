"""GIAANNcmn_inferenceChunks.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common inference Chunks

Opt-in isolated inference chunks and bounded concurrent process scheduling.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from GIAANNcmn_globalDefs import *
import GIAANNcmn_globalDefs as defs


def executeInferenceChunks():
	if(optimisationInferenceChunks):
		validateInferenceChunkConfiguration()
		started = time.perf_counter()
		workerCount = getInferenceWorkerCount()
		run, jobs = prepareInferenceChunks()
		print(inferenceChunksRunLabel, run, flush=True)
		print(inferenceChunksSettingsLabel, inferenceChunksSequenceCount, workerCount, flush=True)
		results = runInferenceChunks(run, jobs, workerCount, started)
		import GIAANNcmn_prediction
		import GIAANNcmn_debug
		GIAANNcmn_prediction.resetInferenceTop1AccuracyCounts()
		for name in inferenceChunksMetricNames:
			setattr(GIAANNcmn_prediction, name, sum(result["metrics"][name] for result in results))
		if(debugPrintTotalInferenceTokens):
			for name in inferenceChunksDebugMetricNames:
				setattr(GIAANNcmn_debug, name, sum(result["debug_metrics"][name] for result in results))
			GIAANNcmn_debug.printTotalInferenceTokens()
		GIAANNcmn_prediction.printInferenceTop1Accuracy(None)
		print(inferenceChunksTimeLabel, time.perf_counter() - started, flush=True)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


def validateInferenceChunkConfiguration():
	if(optimisationInferenceChunks):
		if(executionMode != "inference" or modalityName != "NLP" or useTrainDuringInference or inferenceTrainFirstSequences or inferenceStartGenerateGlobalFeatureNeuronsTensor or not trainLoadExistingDatabase or not useSaveData or not inferencePromptPreserveLineSequenceBoundaries or numberEpochs != inferenceChunksOne or trainSetStartOffsetSequences != inferenceChunksZero or useGPUdense or useGPUsparse or useGPUfileio or (storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and useGPUdatabase) or useAutoresearch or useDrawNetworkIndependently or drawNetworkDuringInference or drawNetworkDuringInferenceSave or inferenceReportGroundedAccuracy or datasetType in closedWorldGroundedDatasetTypes or printInferenceTop1AccuracyBitsPerByte or not printInferenceTop1Accuracy or os.name != inferenceChunksPlatform):
			raise RuntimeError(inferenceChunksErrorConfiguration)
		if(type(inferenceChunksSequenceCount) is not int or inferenceChunksSequenceCount <= inferenceChunksZero or type(inferenceChunksWorkerThreads) is not int or inferenceChunksWorkerThreads != inferenceChunksOne):
			raise ValueError(inferenceChunksErrorParameters)
		for path in (databaseFolder, observedColumnsDir):
			if(not Path(path).is_dir()):
				raise RuntimeError(inferenceChunksErrorDatabase + str(path))
		for path in (conceptColumnsDictFile, conceptFeaturesDictFile):
			if(not Path(path).is_file()):
				raise RuntimeError(inferenceChunksErrorDatabase + str(path))
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


def getInferenceWorkerCount():
	if(optimisationInferenceChunks):
		result = inferenceChunksOne
		if(optimisationInferenceWorkers):
			if(type(inferenceWorkersCount) is not int or inferenceWorkersCount <= inferenceChunksZero):
				raise ValueError(inferenceChunksErrorWorkers)
			result = inferenceWorkersCount
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return result


def prepareInferenceChunks():
	if(optimisationInferenceChunks):
		jobs = []
		import GIAANNnlp_main
		prompt = Path(inferencePromptFile).resolve()
		if(inferenceCopyTemplateDatasets and prompt == (Path(databaseFolder) / inferencePromptFileName).resolve()):
			prompt = (Path(databaseFolderTemplate) / inferencePromptFileName).resolve()
		if(not prompt.is_file()):
			raise RuntimeError(inferenceChunksErrorPrompt + str(prompt))
		promptBytes = prompt.read_bytes()
		lines = [line.lstrip() for line in promptBytes.decode(inferenceChunksEncoding).splitlines() if line.strip()]
		# Match processPrompt's short-line filtering before making independent chunks.
		parsed = list(GIAANNnlp_main.nlpSequence.pipe(lines))
		if(len(parsed) != len(lines) * inferencePromptExpectedSequencesPerLine):
			raise RuntimeError(inferenceChunksErrorPrompt + str(prompt))
		selected = [line for line, sequence in zip(lines, parsed) if len(sequence) >= numSeedTokensInference + inferencePromptExpectedSequencesPerLine]
		if(not selected):
			raise RuntimeError(inferenceChunksErrorPrompt + str(prompt))
		outputRoot = Path(tempfile.gettempdir()).resolve() if inferenceChunksOutputDirectory is None else Path(inferenceChunksOutputDirectory).resolve()
		source = Path(__file__).resolve().parent
		if(not outputRoot.is_dir() or outputRoot.is_relative_to(source.parent) or outputRoot.is_relative_to(Path(databaseFolder).resolve())):
			raise RuntimeError(inferenceChunksErrorOutput)
		run = Path(tempfile.mkdtemp(prefix=inferenceChunksTemporaryPrefix, dir=outputRoot)).resolve()
		snapshot = run / inferenceChunksSourceDirectory
		sourceHashes = snapshotInferenceSource(source, snapshot)
		settings = getInferenceChunkSettings()
		for start in range(inferenceChunksZero, len(selected), inferenceChunksSequenceCount):
			stop = min(start + inferenceChunksSequenceCount, len(selected))
			directory = run / inferenceChunksJobsDirectory / inferenceChunksJobNameFormat.format(len(jobs))
			directory.mkdir(parents=True)
			chunkPrompt = directory / inferenceChunksPromptFilename
			chunkPrompt.write_text(inferenceChunksNewline.join(selected[start:stop]) + inferenceChunksNewline, encoding=inferenceChunksEncoding)
			job = dict(index=len(jobs), start=start, stop=stop, directory=str(directory), working_directory=str(Path.cwd().resolve()), source_directory=str(source), snapshot_directory=str(snapshot), database=str(Path(databaseFolder).resolve()), prompt_sha256=hashlib.sha256(chunkPrompt.read_bytes()).hexdigest(), settings=settings)
			writeInferenceChunkJSON(directory / inferenceChunksConfigFilename, job)
			jobs.append(job)
		manifest = dict(database=str(Path(databaseFolder).resolve()), source_prompt=str(prompt), prompt_sha256=hashlib.sha256(promptBytes).hexdigest(), prompt_lines=len(lines), selected_sequences=len(selected), source_hashes=sourceHashes, settings=settings, operational_overrides=inferenceChunksOperationalOverrides, jobs=[dict(index=job["index"], start=job["start"], stop=job["stop"], directory=job["directory"]) for job in jobs])
		writeInferenceChunkJSON(run / inferenceChunksManifestFilename, manifest)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return run, jobs


def snapshotInferenceSource(source, destination):
	if(optimisationInferenceChunks):
		result = {}
		for path in source.rglob(inferenceChunksGlobPattern):
			if(path.is_symlink()):
				raise RuntimeError(inferenceChunksErrorSource + str(path))
		shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*inferenceChunksIgnorePatterns))
		for path in sorted(destination.rglob(inferenceChunksGlobPattern)):
			if(path.is_file()):
				result[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return result


def getInferenceChunkSettings():
	if(optimisationInferenceChunks):
		result = {name:value for name,value in vars(defs).items() if not name.startswith("_") and (value is None or isinstance(value, (str, int, float, bool)))}
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return result


def runInferenceChunks(run, jobs, workerCount, started):
	if(optimisationInferenceChunks):
		results = []
		if(type(workerCount) is not int or workerCount <= inferenceChunksZero):
			raise ValueError(inferenceChunksErrorWorkers)
		active = {}
		nextJob = inferenceChunksZero
		previousSignal = signal.signal(signal.SIGTERM, interruptInferenceChunks)
		try:
			while(nextJob < len(jobs) or active):
				while(nextJob < len(jobs) and len(active) < workerCount):
					job = jobs[nextJob]
					directory = Path(job["directory"])
					log = (directory / inferenceChunksLogFilename).open("w", encoding=inferenceChunksEncoding)
					environment = dict(os.environ)
					environment.update(inferenceChunksEnvironment)
					environment.pop(inferenceChunksPythonPathEnvironment, None)
					environment[inferenceChunksMatplotlibEnvironment] = str(directory / inferenceChunksCacheDirectory)
					try:
						process = subprocess.Popen([sys.executable, str(Path(job["snapshot_directory"]) / inferenceChunksWorkerFilename), str(directory / inferenceChunksConfigFilename)], cwd=job["working_directory"], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
					except BaseException:
						log.close()
						raise
					active[nextJob] = (process, log, job)
					nextJob += inferenceChunksOne
				writeInferenceChunkSummary(run, jobs, results, active, inferenceChunksStatusRunning, started)
				time.sleep(inferenceChunksPollSeconds)
				for index, (process, log, job) in list(active.items()):
					if(process.poll() is not None):
						log.close()
						del active[index]
						if(process.returncode != inferenceChunksZero):
							raise RuntimeError(inferenceChunksErrorWorker + str(Path(job["directory"]) / inferenceChunksLogFilename))
						results.append(readInferenceChunkResult(job))
						print(inferenceChunksProgressLabel, len(results), "/", len(jobs), flush=True)
			results.sort(key=lambda result: result["index"])
			if([result["index"] for result in results] != list(range(len(jobs))) or sum(result["metrics"][inferenceChunksPredictionTokensMetric] for result in results) <= inferenceChunksZero):
				raise RuntimeError(inferenceChunksErrorCoverage)
			writeInferenceChunkSummary(run, jobs, results, active, inferenceChunksStatusComplete, started)
		except BaseException as error:
			state = inferenceChunksStatusInterrupted if isinstance(error, KeyboardInterrupt) else inferenceChunksStatusFailed
			stopInferenceChunks(active)
			active.clear()
			writeInferenceChunkSummary(run, jobs, results, active, state, started, str(error))
			raise
		finally:
			stopInferenceChunks(active)
			signal.signal(signal.SIGTERM, previousSignal)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return results


def readInferenceChunkResult(job):
	if(optimisationInferenceChunks):
		path = Path(job["directory"]) / inferenceChunksResultFilename
		result = json.loads(path.read_text(encoding=inferenceChunksEncoding))
		if(any(result[key] != job[key] for key in ("index", "start", "stop", "prompt_sha256"))):
			raise RuntimeError(inferenceChunksErrorResult + str(path))
		validateInferenceChunkCounts(result["metrics"])
		if(len(result["sequences"]) > job["stop"] - job["start"]):
			raise RuntimeError(inferenceChunksErrorResult + str(path))
		indices = []
		for sequence in result["sequences"]:
			validateInferenceChunkCounts(sequence["metrics"])
			index = sequence["sequence"]
			if(type(index) is not int or not job["start"] <= index < job["stop"]):
				raise RuntimeError(inferenceChunksErrorResult + str(path))
			indices.append(index)
		if(indices != sorted(set(indices)) or any(sum(sequence["metrics"][name] for sequence in result["sequences"]) != result["metrics"][name] for name in inferenceChunksMetricNames)):
			raise RuntimeError(inferenceChunksErrorResult + str(path))
		if(any(type(result["debug_metrics"][name]) is not int or result["debug_metrics"][name] < inferenceChunksZero for name in inferenceChunksDebugMetricNames)):
			raise RuntimeError(inferenceChunksErrorResult + str(path))
		if(type(result["elapsed"]) not in (int, float) or not math.isfinite(result["elapsed"]) or result["elapsed"] < inferenceChunksZero or type(result["rss_kib"]) is not int or result["rss_kib"] <= inferenceChunksZero):
			raise RuntimeError(inferenceChunksErrorResult + str(path))
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return result


def validateInferenceChunkCounts(metrics):
	if(optimisationInferenceChunks):
		if(set(metrics) != set(inferenceChunksMetricNames) or any(type(value) is not int or value < inferenceChunksZero for value in metrics.values())):
			raise RuntimeError(inferenceChunksErrorResult + str(metrics))
		matches, tokens, predictionMatches, predictionTokens = (metrics[name] for name in inferenceChunksMetricNames)
		if(not inferenceChunksZero <= predictionMatches <= predictionTokens <= tokens or not predictionMatches <= matches <= tokens or matches - predictionMatches > tokens - predictionTokens):
			raise RuntimeError(inferenceChunksErrorResult + str(metrics))
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


def writeInferenceChunkSummary(run, jobs, results, active, state, started, error=None):
	if(optimisationInferenceChunks):
		metrics = {name:sum(result["metrics"][name] for result in results) for name in inferenceChunksMetricNames}
		predictionTokens = metrics[inferenceChunksPredictionTokensMetric]
		tokens = metrics[inferenceChunksTokensMetric]
		summary = dict(status=state, completed_chunks=len(results), total_chunks=len(jobs), active_workers={index:process.pid for index,(process,log,job) in active.items()}, metrics=metrics, prediction_accuracy=metrics[inferenceChunksPredictionMatchesMetric]/predictionTokens if predictionTokens else None, inference_accuracy=metrics[inferenceChunksMatchesMetric]/tokens if tokens else None, elapsed_wall_seconds=time.perf_counter()-started, elapsed_worker_seconds=sum(result["elapsed"] for result in results), peak_worker_rss_kib=max((result["rss_kib"] for result in results), default=inferenceChunksZero), error=error)
		writeInferenceChunkJSON(run / inferenceChunksSummaryFilename, summary)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


def stopInferenceChunks(active):
	if(optimisationInferenceChunks):
		for process, log, job in active.values():
			if(process.poll() is None):
				try:
					os.killpg(process.pid, signal.SIGTERM)
				except ProcessLookupError:
					pass	#The process exited between poll and signal; wait below still reaps it.
		deadline = time.monotonic() + inferenceChunksStopSeconds
		for process, log, job in active.values():
			try:
				process.wait(timeout=max(inferenceChunksZero, deadline - time.monotonic()))
			except subprocess.TimeoutExpired:
				try:
					os.killpg(process.pid, signal.SIGKILL)
				except ProcessLookupError:
					pass	#The process exited after the timeout; wait below still reaps it.
				process.wait()
			log.close()
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return


def interruptInferenceChunks(number, frame):
	if(optimisationInferenceChunks):
		raise KeyboardInterrupt()
	return


def writeInferenceChunkJSON(path, value):
	if(optimisationInferenceChunks):
		temporary = path.with_name(path.name + inferenceChunksTemporaryExtension)
		temporary.write_text(json.dumps(value, indent=inferenceChunksJSONIndent, allow_nan=False) + inferenceChunksNewline, encoding=inferenceChunksEncoding)
		os.replace(temporary, path)
	else:
		raise RuntimeError(inferenceChunksErrorDisabled)
	return
