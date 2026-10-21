#!/usr/bin/env python3
"""Prepare isolated source configurations and schedule resumable inference jobs."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import math
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from generalisedBenchmarkTool_globalDefs import *
from generalisedBenchmarkToolFiles import fileHashes, hashBytes, prepareConfiguration, readJSON, snapshotRepository, verifyHashes, writeJSON


def main():
	if(generalisedBenchmarkTool):
		args = parseArguments()
		if(args.resume is None):
			run = prepareRun(args)
		else:
			if(any(value is not None for value in (args.repository, args.configurations, args.runsDirectory, args.python, args.numConfigurations, args.sequencesPerJob, args.maxSequences))):
				raise ValueError(benchmarkErrorResume)
			run = Path(args.resume).resolve()
		manifest = readJSON(run / benchmarkManifestFilename)
		validateRun(run, manifest)
		if(not args.prepareOnly):
			with (run / benchmarkLockFilename).open('a') as lock:
				try:
					fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
				except BlockingIOError as error:
					raise RuntimeError(benchmarkErrorLock) from error
				runJobs(run, manifest, args.numWorkers, lock.fileno())
	else:
		raise RuntimeError(benchmarkErrorDisabled)
	return


def parseArguments():
	args = None
	if(generalisedBenchmarkTool):
		parser = argparse.ArgumentParser(description=benchmarkDescription)
		argumentTypes = {benchmarkArgumentPositive: positiveInteger, benchmarkArgumentNonnegative: nonnegativeInteger, benchmarkArgumentPath: Path}
		for flag, definition in benchmarkArguments:
			options = dict(definition)
			if(benchmarkArgumentKindKey in options):
				options[benchmarkArgumentTypeKey] = argumentTypes[options.pop(benchmarkArgumentKindKey)]
			parser.add_argument(flag, **options)
		args = parser.parse_args()
	return args


def prepareRun(args):
	run = None
	if(generalisedBenchmarkTool):
		repository = (args.repository or benchmarkRepository).resolve()
		configurations = (args.configurations or benchmarkDirectory / benchmarkConfigurationsDirectory).resolve()
		runsDirectory = (args.runsDirectory or benchmarkDirectory / benchmarkRunsDirectory).resolve()
		python = (args.python or benchmarkPython).resolve()
		if(runsDirectory.is_relative_to(repository) or runsDirectory.is_relative_to(configurations)):
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
		snapshotRepository(repository, run / benchmarkSnapshotDirectory)
		entries = []
		for configuration in files:
			directory = run / benchmarkConfigurationsDirectory / configuration.stem
			prepareConfiguration(run / benchmarkSnapshotDirectory, configuration, directory)
			request = dict(configuration=configuration.stem, repository=str(repository), configuration_directory=str(directory), max_sequences=args.maxSequences, sequences_per_job=benchmarkDefaultSequencesPerJob if args.sequencesPerJob is None else args.sequencesPerJob)
			writeJSON(directory / benchmarkJobFilename, request)
			entries.append(dict(name=configuration.stem, directory=str(directory)))
		jobs = []
		for entry in entries:
			directory = Path(entry['directory'])
			logPath = directory / benchmarkPrepareLogFilename
			with logPath.open('w', encoding=benchmarkEncoding) as log:
				completed = subprocess.run([str(python), str(engine / benchmarkWorkerFilename), benchmarkPrepareMode, str(directory / benchmarkJobFilename)], env=workerEnvironment(), stdout=log, stderr=subprocess.STDOUT, check=False)
			if(completed.returncode != benchmarkExitSuccess):
				raise RuntimeError(benchmarkErrorWorker + str(logPath))
			plan = readJSON(directory / benchmarkPlanFilename)
			jobs.extend(prepareJobs(run, directory, plan, len(jobs)))
			entry['hashes'] = fileHashes(directory)
		manifest = dict(schema=benchmarkSchemaVersion, prepared=True, created=datetime.now(timezone.utc).isoformat(), repository=str(repository), python=str(python), engine_hashes=fileHashes(engine), source_hashes=fileHashes(run / benchmarkSnapshotDirectory), configurations=entries, jobs=jobs)
		writeJSON(run / benchmarkManifestFilename, manifest)
		writeSummary(run, manifest, benchmarkStatusPrepared, {})
		print(benchmarkMessagePrepared + str(run), flush=True)
	return run


def prepareJobs(run, directory, plan, offset):
	jobs = []
	if(generalisedBenchmarkTool):
		count = plan['selected_prompt_lines']
		size = plan['sequences_per_job'] or count
		for start in range(benchmarkZero, count, size):
			identifier = benchmarkJobNameFormat.format(offset + len(jobs) + benchmarkOne)
			jobDirectory = run / benchmarkJobsDirectory / identifier
			jobDirectory.mkdir(parents=True)
			jobFile = jobDirectory / benchmarkJobFilename
			job = dict(id=identifier, configuration=plan['configuration'], configuration_directory=str(directory), repository=plan['repository'], output=str(jobDirectory / benchmarkOutputDirectory), job_file=str(jobFile), start=start, stop=min(start+size, count), sliced=plan['sequences_per_job'] > benchmarkZero or plan['max_sequences'] is not None, max_sequences=plan['max_sequences'], sequences_per_job=plan['sequences_per_job'])
			writeJSON(jobFile, job)
			jobs.append(dict(id=identifier, configuration=plan['configuration'], file=str(jobFile), sha256=hashBytes(jobFile.read_bytes())))
	return jobs


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
		for entry in manifest['jobs']:
			if(hashBytes(Path(entry['file']).read_bytes()) != entry['sha256']):
				raise RuntimeError(benchmarkErrorChanged + entry['file'])
	return


def runJobs(run, manifest, numWorkers, lockDescriptor):
	if(generalisedBenchmarkTool):
		active = {}
		completed = {}
		pending = []
		for entry in manifest['jobs']:
			job = readJSON(entry['file'])
			resultPath = Path(job['output']) / benchmarkResultFilename
			if(resultPath.exists()):
				completed[entry['id']] = validatedResult(entry, job)
			else:
				pending.append(entry)
		pending.sort(key=lambda entry: (readJSON(entry['file'])['start'], entry['configuration']))
		oldHandler = signal.signal(signal.SIGTERM, interrupt)
		try:
			while(pending or active):
				while(pending and len(active) < numWorkers):
					entry = pending.pop(benchmarkZero)
					job = readJSON(entry['file'])
					output = Path(job['output'])
					if(output.exists()):
						history = output.parent / benchmarkHistoryDirectory
						history.mkdir(exist_ok=True)
						output.rename(history / datetime.now(timezone.utc).strftime(benchmarkDateFormat))
					output.mkdir()
					log = (output / benchmarkLogFilename).open('w', encoding=benchmarkEncoding)
					try:
						process = subprocess.Popen([manifest['python'], str(run / benchmarkEngineDirectory / benchmarkWorkerFilename), benchmarkRunMode, entry['file']], env=workerEnvironment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lockDescriptor,))
					except BaseException:
						log.close()
						raise
					active[entry['id']] = (process, log, entry, job)
					print(benchmarkMessageStarted + entry['id'] + ' ' + entry['configuration'], flush=True)
				writeSummary(run, manifest, benchmarkStatusRunning, completed, active)
				time.sleep(benchmarkPollSeconds)
				for identifier, (process, log, entry, job) in list(active.items()):
					if(process.poll() is not None):
						log.close()
						del active[identifier]
						if(process.returncode != benchmarkExitSuccess):
							raise RuntimeError(benchmarkErrorWorker + str(Path(job['output']) / benchmarkLogFilename) + ' (exit ' + str(process.returncode) + ')')
						completed[identifier] = validatedResult(entry, job)
						print(benchmarkMessageFinished + identifier + ' ' + str(len(completed)) + '/' + str(len(manifest['jobs'])), flush=True)
			writeSummary(run, manifest, benchmarkStatusComplete, completed)
			print(benchmarkMessageComplete + str(run), flush=True)
		except BaseException as error:
			stopWorkers(active)
			state = benchmarkStatusInterrupted if isinstance(error, KeyboardInterrupt) else benchmarkStatusFailed
			writeSummary(run, manifest, state, completed, {}, str(error))
			print(benchmarkMessageResume + str(benchmarkDirectory / benchmarkDriverFilename) + ' -numWorkers ' + str(numWorkers) + ' -resume ' + str(run), flush=True)
			raise
		finally:
			signal.signal(signal.SIGTERM, oldHandler)
	return


def stopWorkers(active):
	if(generalisedBenchmarkTool):
		for process, log, entry, job in active.values():
			if(process.poll() is None):
				os.killpg(process.pid, signal.SIGTERM)
		deadline = time.monotonic() + benchmarkStopGraceSeconds
		for process, log, entry, job in active.values():
			try:
				process.wait(timeout=max(benchmarkZero, deadline - time.monotonic()))
			except subprocess.TimeoutExpired:
				os.killpg(process.pid, signal.SIGKILL)
				process.wait()
			log.close()
	return


def validatedResult(entry, job):
	result = None
	if(generalisedBenchmarkTool):
		path = Path(job['output']) / benchmarkResultFilename
		result = readJSON(path)
		for key in ('matches', 'tokens', 'sequences'):
			if(type(result[key]) is not int or result[key] < benchmarkZero):
				raise RuntimeError(benchmarkErrorResult + str(path))
		if(result['tokens'] <= benchmarkZero or result['sequences'] <= benchmarkZero or not benchmarkZero <= result['matches'] <= result['tokens'] or result['accuracy'] != result['matches']/result['tokens'] or result['job_sha256'] != entry['sha256'] or result['job_id'] != entry['id'] or result['configuration'] != entry['configuration']):
			raise RuntimeError(benchmarkErrorResult + str(path))
		if(not math.isfinite(result['elapsed']) or result['elapsed'] < benchmarkZero or result['rss_kib'] <= benchmarkZero):
			raise RuntimeError(benchmarkErrorResult + str(path))
		plan = readJSON(Path(job['configuration_directory']) / benchmarkPlanFilename)
		if(any(result[key] != plan[key] for key in ('database', 'prompt_sha256', 'sampled')) or any(result[key] != job[key] for key in ('start', 'stop'))):
			raise RuntimeError(benchmarkErrorResult + str(path))
		rows = [json.loads(line) for line in (Path(job['output']) / benchmarkSequencesFilename).read_text(encoding=benchmarkEncoding).splitlines()]
		if(len(rows) != result['sequences'] or sum(row['matches'] for row in rows) != result['matches'] or sum(row['tokens'] for row in rows) != result['tokens']):
			raise RuntimeError(benchmarkErrorResult + str(path))
	return result


def writeSummary(run, manifest, state, completed, active=None, error=None):
	if(generalisedBenchmarkTool):
		rows = []
		for configuration in manifest['configurations']:
			name = configuration['name']
			entries = [entry for entry in manifest['jobs'] if entry['configuration'] == name]
			results = [completed[entry['id']] for entry in entries if entry['id'] in completed]
			plan = readJSON(Path(configuration['directory']) / benchmarkPlanFilename)
			tokens = sum(result['tokens'] for result in results)
			matches = sum(result['matches'] for result in results)
			status = benchmarkStatusComplete if len(results) == len(entries) else state
			rows.append(dict(configuration=name, status=status, completed_jobs=len(results), total_jobs=len(entries), sequences=sum(result['sequences'] for result in results), matches=matches, tokens=tokens, accuracy_percent=benchmarkPercent*matches/tokens if tokens else None, elapsed_worker_seconds=sum(result['elapsed'] for result in results), peak_worker_rss_kib=max((result['rss_kib'] for result in results), default=benchmarkZero), sampled=plan['sampled'], database=plan['database'], prompt_sha256=plan['prompt_sha256']))
		writeJSON(run / benchmarkSummaryFilename, dict(status=state, updated=datetime.now(timezone.utc).isoformat(), completed_jobs=len(completed), total_jobs=len(manifest['jobs']), active_jobs={identifier:process.pid for identifier,(process,log,entry,job) in (active or {}).items()}, error=error, configurations=rows))
		temporary = run / (benchmarkSummaryCSVFilename + benchmarkTemporaryExtension)
		with temporary.open('w', encoding=benchmarkEncoding, newline='') as stream:
			writer = csv.DictWriter(stream, fieldnames=benchmarkCSVFields)
			writer.writeheader()
			writer.writerows(rows)
		os.replace(temporary, run / benchmarkSummaryCSVFilename)
	return


def workerEnvironment():
	result = None
	if(generalisedBenchmarkTool):
		result = dict(os.environ)
		result.pop('PYTHONPATH', None)
		result.update(benchmarkEnvironment)
	return result


def positiveInteger(value):
	result = int(value)
	if(generalisedBenchmarkTool):
		if(result <= benchmarkZero):
			raise argparse.ArgumentTypeError(benchmarkErrorPositive)
	return result


def nonnegativeInteger(value):
	result = int(value)
	if(generalisedBenchmarkTool):
		if(result < benchmarkZero):
			raise argparse.ArgumentTypeError(benchmarkErrorNonnegative)
	return result


def interrupt(number, frame):
	if(generalisedBenchmarkTool):
		raise KeyboardInterrupt()
	return


if(__name__ == '__main__'):
	main()
