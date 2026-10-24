"""Inspect or execute one frozen configuration through the native proto entry point."""
import importlib
import os
from pathlib import Path
import signal
import sys
import time
from generalisedBenchmarkTool_globalDefs import *
from generalisedBenchmarkToolFiles import fileHashes, hashBytes, readJSON, writeJSON


def main():
	if(generalisedBenchmarkTool):
		if(len(sys.argv) != benchmarkConfigurationArgumentCount):
			raise ValueError(benchmarkErrorArguments)
		mode, specification = sys.argv[benchmarkOne:]
		config = readJSON(specification)
		directory = Path(config['directory'])
		proto = directory / benchmarkSnapshotDirectory / benchmarkProtoDirectory
		sys.dont_write_bytecode = True
		sys.path.insert(benchmarkZero, str(proto))
		os.chdir(proto)
		signal.signal(signal.SIGTERM, interrupt)
		defs = importlib.import_module(benchmarkCommonModule)
		chunks = importlib.import_module(benchmarkChunksModule)
		validateSettings(defs, chunks)
		nativeWorker = importlib.import_module(benchmarkNativeWorkerModule)
		nativeWorker.installInferenceChunkWriteProtection([Path(config['repository']).resolve()], [directory.resolve(), Path(config['output']).resolve()])
		nativeWorker.installInferenceChunkWriteProtection([(directory / benchmarkSnapshotDirectory).resolve(), Path(defs.databaseFolder).resolve()])
		if(mode == benchmarkPrepareMode):
			inspectConfiguration(config, defs, chunks)
		elif(mode == benchmarkValidateMode):
			if(config['num_workers'] is not None and (not defs.optimisationInferenceWorkers or chunks.getInferenceWorkerCount() != config['num_workers'])):
				raise ValueError(benchmarkErrorConfiguration + benchmarkWorkersSetting)
			writeJSON(directory / benchmarkSettingsFilename, chunks.getInferenceChunkSettings())
		elif(mode == benchmarkRunMode):
			runConfiguration(config, defs, chunks)
		else:
			raise ValueError(benchmarkErrorConfiguration + mode)
	else:
		raise RuntimeError(benchmarkErrorDisabled)
	return


def inspectConfiguration(config, defs, chunks):
	if(generalisedBenchmarkTool):
		directory = Path(config['directory'])
		database = Path(defs.databaseFolder).resolve()
		prompt = Path(defs.inferencePromptFile).resolve()
		if(defs.inferenceCopyTemplateDatasets and prompt == (database / defs.inferencePromptFileName).resolve()):
			prompt = (Path(defs.databaseFolderTemplate) / defs.inferencePromptFileName).resolve()
		if(not prompt.is_file()):
			raise ValueError(benchmarkErrorPrompt + str(prompt))
		data = prompt.read_bytes()
		lines = [line for line in data.decode(benchmarkEncoding).splitlines() if line.strip()]
		if(not lines):
			raise ValueError(benchmarkErrorPrompt + str(prompt))
		count = len(lines) if config['max_sequences'] is None else config['max_sequences']
		if(count > len(lines)):
			raise ValueError(benchmarkErrorPartial)
		selected = data if count == len(lines) else (benchmarkNewline.join(lines[:count]) + benchmarkNewline).encode(benchmarkEncoding)
		(directory / benchmarkPromptFilename).write_bytes(selected)
		writeJSON(directory / benchmarkOriginalSettingsFilename, chunks.getInferenceChunkSettings())
		plan = dict(configuration=config['name'], database=str(database), source_prompt=str(prompt), prompt_sha256=hashBytes(data), selected_prompt_sha256=hashBytes(selected), prompt_lines=len(lines), selected_prompt_lines=count, sampled=count != len(lines))
		writeJSON(directory / benchmarkPlanFilename, plan)
	return


def runConfiguration(config, defs, chunks):
	if(generalisedBenchmarkTool):
		directory = Path(config['directory'])
		plan = readJSON(directory / benchmarkPlanFilename)
		settings = readJSON(directory / benchmarkSettingsFilename)
		if(chunks.getInferenceChunkSettings() != settings or str(Path(defs.databaseFolder).resolve()) != plan['database'] or hashBytes((directory / benchmarkPromptFilename).read_bytes()) != plan['selected_prompt_sha256']):
			raise RuntimeError(benchmarkErrorChanged + str(directory))
		started = time.perf_counter()
		import torch
		mainModule = importlib.import_module(benchmarkMainModule)
		with torch.no_grad():
			mainModule.main()
		nativeRoot = Path(config['native_directory'])
		runs = list(nativeRoot.iterdir())
		if(len(runs) != benchmarkOne or not runs[benchmarkZero].is_dir()):
			raise RuntimeError(benchmarkErrorNativeRun + str(nativeRoot))
		nativeRun = runs[benchmarkZero]
		summary = readJSON(nativeRun / defs.inferenceChunksSummaryFilename)
		manifest = readJSON(nativeRun / defs.inferenceChunksManifestFilename)
		if(summary['status'] != defs.inferenceChunksStatusComplete or summary['completed_chunks'] != summary['total_chunks'] or summary['active_workers'] or summary['error'] is not None or summary['total_chunks'] != len(manifest['jobs'])):
			raise RuntimeError(benchmarkErrorResult + str(nativeRun))
		if(manifest['settings'] != settings or manifest['database'] != plan['database'] or manifest['prompt_sha256'] != plan['selected_prompt_sha256']):
			raise RuntimeError(benchmarkErrorChanged + str(nativeRun))
		chunks.validateInferenceChunkCounts(summary['metrics'])
		sequences = summary['sequences']
		metrics = summary['metrics']
		if(metrics[benchmarkNativePredictionTokens] <= benchmarkZero or sequences <= benchmarkZero):
			raise RuntimeError(benchmarkErrorNoTokens)
		result = dict(configuration=config['name'], database=plan['database'], prompt_sha256=plan['prompt_sha256'], selected_prompt_sha256=plan['selected_prompt_sha256'], sampled=plan['sampled'], metrics=metrics, sequences=sequences, matches=metrics[benchmarkNativePredictionMatches], tokens=metrics[benchmarkNativePredictionTokens], accuracy=summary['prediction_accuracy'], inference_accuracy=summary['inference_accuracy'], elapsed_wall_seconds=time.perf_counter()-started, elapsed_worker_seconds=summary['elapsed_worker_seconds'], peak_worker_rss_kib=summary['peak_worker_rss_kib'], native_run=str(nativeRun), native_hashes=fileHashes(nativeRun))
		writeJSON(Path(config['output']) / benchmarkResultFilename, result)
	return


def validateSettings(defs, chunks):
	if(generalisedBenchmarkTool):
		for name, value in benchmarkRequiredSettings.items():
			if(getattr(defs, name, None) != value):
				raise ValueError(benchmarkErrorConfiguration + name + '=' + repr(getattr(defs, name, None)))
		chunks.validateInferenceChunkConfiguration()
		chunks.getInferenceWorkerCount()
	return


def interrupt(number, frame):
	if(generalisedBenchmarkTool):
		raise KeyboardInterrupt()
	return


if(__name__ == '__main__'):
	main()
