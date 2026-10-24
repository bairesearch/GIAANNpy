#!/usr/bin/env python3
"""Run isolated configurations sequentially, delegating inference parallelism to proto."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
from generalisedBenchmarkTool_globalDefs import *
from generalisedBenchmarkToolFiles import applyExecutionSettings, fileHashes, prepareConfiguration, readJSON, snapshotRepository, verifyHashes, writeJSON


def main():
	if(generalisedBenchmarkTool):
		args = parseArguments()
		if(args.resume is None):
			run = prepareRun(args)
		else:
			if(any(value is not None for value in (args.repository, args.configurations, args.runsDirectory, args.python, args.numConfigurations, args.numWorkers, args.sequencesPerJob, args.maxSequences))):
				raise ValueError(benchmarkErrorResume)
			run = args.resume.resolve()
		manifest = readJSON(run / benchmarkManifestFilename)
		validateRun(run, manifest)
		if(not args.prepareOnly):
			with (run / benchmarkLockFilename).open('a') as lock:
				try:
					fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
				except BlockingIOError as error:
					raise RuntimeError(benchmarkErrorLock) from error
				runConfigurations(run, manifest, lock.fileno())
	else:
		raise RuntimeError(benchmarkErrorDisabled)
	return


def parseArguments():
	if(generalisedBenchmarkTool):
		parser = argparse.ArgumentParser(description=benchmarkDescription)
		argumentTypes = {benchmarkArgumentPositive: positiveInteger, benchmarkArgumentPath: Path}
		for flag, definition in benchmarkArguments:
			options = dict(definition)
			if(benchmarkArgumentKindKey in options):
				options[benchmarkArgumentTypeKey] = argumentTypes[options.pop(benchmarkArgumentKindKey)]
			parser.add_argument(flag, **options)
		args = parser.parse_args()
	return args


def prepareRun(args):
	if(generalisedBenchmarkTool):
		repository = (args.repository or benchmarkRepository).resolve()
		configurations = (args.configurations or benchmarkDirectory / benchmarkConfigurationsDirectory).resolve()
		runsDirectory = (args.runsDirectory or benchmarkRunsDirectory).resolve()
		python = (args.python or benchmarkPython).resolve()
		if(repository.is_relative_to(runsDirectory) or configurations.is_relative_to(runsDirectory) or runsDirectory.is_relative_to(configurations) or runsDirectory.is_relative_to(repository / benchmarkProtoDirectory) or (runsDirectory.is_relative_to(repository) and benchmarkForbiddenParts.intersection(runsDirectory.relative_to(repository).parts))):
			raise ValueError(benchmarkErrorLocation)
		if(not python.is_file() or not os.access(python, os.X_OK)):
			raise ValueError(benchmarkErrorPath + str(python))
		files = sorted(configurations.glob(benchmarkConfigurationPattern))
		if(not files):
			raise ValueError(benchmarkErrorEmpty)
		if(args.numConfigurations is not None):
			if(args.numConfigurations > len(files)):
				raise ValueError(benchmarkErrorRange + str(args.numConfigurations))
			files = files[:args.numConfigurations]
		run = runsDirectory / datetime.now(timezone.utc).strftime(benchmarkDateFormat)
		run.mkdir(parents=True)
		engine = run / benchmarkEngineDirectory
		engine.mkdir()
		for filename in benchmarkEngineFilenames:
			shutil.copy2(benchmarkDirectory / filename, engine / filename)
		snapshotRepository(repository, run / benchmarkSnapshotDirectory, (runsDirectory, benchmarkRunsDirectory, repository / benchmarkRepositoryRunsPath))
		entries = []
		for configuration in files:
			directory = run / benchmarkConfigurationsDirectory / configuration.stem
			output = run / benchmarkOutputDirectory / configuration.stem
			output.mkdir(parents=True)
			prepareConfiguration(run / benchmarkSnapshotDirectory, configuration, directory)
			entry = dict(name=configuration.stem, repository=str(repository), directory=str(directory), output=str(output), native_directory=str(output / benchmarkNativeDirectory), max_sequences=args.maxSequences, num_workers=args.numWorkers, sequences_per_job=args.sequencesPerJob)
			writeJSON(directory / benchmarkSpecificationFilename, entry)
			configurationProcess(run, str(python), entry, benchmarkPrepareMode, output / benchmarkPrepareLogFilename)
			overrides = dict(benchmarkRuntimeOverrides)
			overrides[benchmarkPromptSetting] = str(directory / benchmarkPromptFilename)
			overrides[benchmarkNativeOutputSetting] = entry['native_directory']
			if(args.numWorkers is not None):
				overrides[benchmarkWorkerCountSetting] = args.numWorkers
			if(args.sequencesPerJob is not None):
				overrides[benchmarkChunkSizeSetting] = args.sequencesPerJob
			applyExecutionSettings(directory / benchmarkSnapshotDirectory, overrides)
			plan = readJSON(directory / benchmarkPlanFilename)
			plan['runtime_overrides'] = overrides
			writeJSON(directory / benchmarkPlanFilename, plan)
			configurationProcess(run, str(python), entry, benchmarkValidateMode, output / benchmarkValidateLogFilename)
			entry['hashes'] = fileHashes(directory)
			entries.append(entry)
		manifest = dict(schema=benchmarkSchemaVersion, prepared=True, created=datetime.now(timezone.utc).isoformat(), repository=str(repository), python=str(python), engine_hashes=fileHashes(engine), source_hashes=fileHashes(run / benchmarkSnapshotDirectory), configurations=entries)
		writeJSON(run / benchmarkManifestFilename, manifest)
		writeSummary(run, manifest, benchmarkStatusPrepared, {})
		print(benchmarkMessagePrepared + str(run), flush=True)
	return run


def validateRun(run, manifest):
	if(generalisedBenchmarkTool):
		if(manifest['schema'] != benchmarkSchemaVersion):
			raise RuntimeError(benchmarkErrorVersion)
		if(not manifest['prepared']):
			raise RuntimeError(benchmarkErrorIncomplete)
		verifyHashes(run / benchmarkEngineDirectory, manifest['engine_hashes'])
		verifyHashes(run / benchmarkSnapshotDirectory, manifest['source_hashes'])
		for configuration in manifest['configurations']:
			verifyHashes(Path(configuration['directory']), configuration['hashes'])
	return


def runConfigurations(run, manifest, lockDescriptor):
	if(generalisedBenchmarkTool):
		completed = {}
		previousSignal = signal.signal(signal.SIGTERM, interrupt)
		try:
			for entry in manifest['configurations']:
				output = Path(entry['output'])
				if((output / benchmarkResultFilename).is_file()):
					completed[entry['name']] = validatedResult(entry)
			for entry in manifest['configurations']:
				if(entry['name'] not in completed):
					output = Path(entry['output'])
					if(Path(entry['native_directory']).exists()):
						checkInactiveConfiguration(entry)
						history = run / benchmarkHistoryDirectory / entry['name']
						history.mkdir(parents=True, exist_ok=True)
						output.rename(history / datetime.now(timezone.utc).strftime(benchmarkDateFormat))
					Path(entry['native_directory']).mkdir(parents=True)
					writeSummary(run, manifest, benchmarkStatusRunning, completed, entry['name'])
					print(benchmarkMessageStarted + entry['name'] + ' ' + str(output / benchmarkLogFilename), flush=True)
					configurationProcess(run, manifest['python'], entry, benchmarkRunMode, output / benchmarkLogFilename, lockDescriptor)
					completed[entry['name']] = validatedResult(entry)
					print(benchmarkMessageFinished + entry['name'] + ' ' + str(len(completed)) + '/' + str(len(manifest['configurations'])), flush=True)
					writeSummary(run, manifest, benchmarkStatusRunning, completed)
			writeSummary(run, manifest, benchmarkStatusComplete, completed)
			print(benchmarkMessageComplete + str(run), flush=True)
		except BaseException as error:
			state = benchmarkStatusInterrupted if isinstance(error, KeyboardInterrupt) else benchmarkStatusFailed
			writeSummary(run, manifest, state, completed, error=str(error))
			print(benchmarkMessageResume + str(benchmarkDirectory / benchmarkDriverFilename) + ' -resume ' + str(run), flush=True)
			raise
		finally:
			signal.signal(signal.SIGTERM, previousSignal)
	return


def configurationProcess(run, python, entry, mode, logPath, lockDescriptor=None):
	if(generalisedBenchmarkTool):
		environment = dict(os.environ)
		environment.pop(benchmarkPythonPathEnvironmentKey, None)
		environment.update(benchmarkEnvironment)
		environment[benchmarkMatplotlibEnvironmentKey] = str(Path(entry['output']) / benchmarkRuntimeCacheDirectory)
		descriptors = () if lockDescriptor is None else (lockDescriptor,)
		with logPath.open(benchmarkLogMode, encoding=benchmarkEncoding) as log:
			process = subprocess.Popen([python, str(run / benchmarkEngineDirectory / benchmarkConfigurationRunnerFilename), mode, str(Path(entry['directory']) / benchmarkSpecificationFilename)], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=descriptors)
			try:
				code = process.wait()
			except BaseException:
				if(process.poll() is None):
					process.terminate()
				try:
					process.wait(timeout=benchmarkStopGraceSeconds)
				except subprocess.TimeoutExpired as error:
					raise RuntimeError(benchmarkErrorCleanup + str(process.pid)) from error
				raise
		if(code != benchmarkExitSuccess):
			raise RuntimeError(benchmarkErrorProcess + str(logPath))
	return


def checkInactiveConfiguration(entry):
	if(generalisedBenchmarkTool):
		for path in Path(entry['native_directory']).glob(benchmarkFileGlob + '/' + benchmarkSummaryFilename):
			summary = readJSON(path)
			for pid in summary['active_workers'].values():
				try:
					os.kill(pid, benchmarkZero)
				except ProcessLookupError:
					pass	#This recorded worker has already exited.
				else:
					raise RuntimeError(benchmarkErrorActive + str(pid))
	return


def validatedResult(entry):
	if(generalisedBenchmarkTool):
		path = Path(entry['output']) / benchmarkResultFilename
		result = readJSON(path)
		plan = readJSON(Path(entry['directory']) / benchmarkPlanFilename)
		if(result['configuration'] != entry['name'] or any(result[key] != plan[key] for key in ('database', 'prompt_sha256', 'selected_prompt_sha256', 'sampled'))):
			raise RuntimeError(benchmarkErrorResult + str(path))
		for name in ('matches', 'tokens', 'sequences', 'peak_worker_rss_kib'):
			if(type(result[name]) is not int or result[name] < benchmarkZero):
				raise RuntimeError(benchmarkErrorResult + str(path))
		if(result['tokens'] <= benchmarkZero or result['sequences'] <= benchmarkZero or result['peak_worker_rss_kib'] <= benchmarkZero or not benchmarkZero <= result['matches'] <= result['tokens'] or result['accuracy'] != result['matches']/result['tokens']):
			raise RuntimeError(benchmarkErrorResult + str(path))
		for name in ('elapsed_wall_seconds', 'elapsed_worker_seconds'):
			if(type(result[name]) not in (int, float) or not math.isfinite(result[name]) or result[name] < benchmarkZero):
				raise RuntimeError(benchmarkErrorResult + str(path))
		nativeRun = Path(result['native_run'])
		if(nativeRun.parent != Path(entry['native_directory'])):
			raise RuntimeError(benchmarkErrorResult + str(path))
		verifyHashes(nativeRun, result['native_hashes'])
		native = readJSON(nativeRun / benchmarkSummaryFilename)
		if(native['status'] != benchmarkStatusComplete or native['sequences'] != result['sequences'] or native['metrics'] != result['metrics'] or native['prediction_accuracy'] != result['accuracy'] or native['inference_accuracy'] != result['inference_accuracy'] or result['metrics'][benchmarkNativePredictionTokens] != result['tokens'] or result['metrics'][benchmarkNativePredictionMatches] != result['matches']):
			raise RuntimeError(benchmarkErrorResult + str(path))
	return result


def writeSummary(run, manifest, state, completed, active=None, error=None):
	if(generalisedBenchmarkTool):
		rows = []
		for entry in manifest['configurations']:
			plan = readJSON(Path(entry['directory']) / benchmarkPlanFilename)
			result = completed.get(entry['name'])
			row = dict.fromkeys(benchmarkCSVFields)
			row.update(configuration=entry['name'], status=benchmarkStatusComplete if result is not None else state, sampled=plan['sampled'], database=plan['database'], prompt_sha256=plan['prompt_sha256'])
			if(result is not None):
				row.update({name:result[name] for name in ('sequences', 'matches', 'tokens', 'elapsed_wall_seconds', 'elapsed_worker_seconds', 'peak_worker_rss_kib', 'native_run')})
				row.update(accuracy_percent=benchmarkPercent*result['accuracy'], inference_accuracy_percent=benchmarkPercent*result['inference_accuracy'])
			rows.append(row)
		writeJSON(run / benchmarkSummaryFilename, dict(status=state, updated=datetime.now(timezone.utc).isoformat(), completed_configurations=len(completed), total_configurations=len(manifest['configurations']), active_configuration=active, error=error, configurations=rows))
		temporary = run / (benchmarkSummaryCSVFilename + benchmarkTemporaryExtension)
		with temporary.open('w', encoding=benchmarkEncoding, newline='') as stream:
			writer = csv.DictWriter(stream, fieldnames=benchmarkCSVFields)
			writer.writeheader()
			writer.writerows(rows)
		os.replace(temporary, run / benchmarkSummaryCSVFilename)
	return


def positiveInteger(value):
	if(generalisedBenchmarkTool):
		result = int(value)
		if(result <= benchmarkZero):
			raise argparse.ArgumentTypeError(benchmarkErrorPositive)
	return result


def interrupt(number, frame):
	if(generalisedBenchmarkTool):
		raise KeyboardInterrupt()
	return


if(__name__ == '__main__'):
	main()
