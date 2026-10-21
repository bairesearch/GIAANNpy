"""One isolated process for configuration inspection or an inference job."""
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from generalisedBenchmarkTool_globalDefs import *
from generalisedBenchmarkToolFiles import hashBytes, readJSON, writeJSON


def main():
	if(generalisedBenchmarkTool):
		mode, specification = sys.argv[benchmarkOne:]
		config = readJSON(specification)
		configurationDirectory = Path(config['configuration_directory'])
		proto = configurationDirectory / benchmarkSnapshotDirectory / benchmarkProtoDirectory
		sys.dont_write_bytecode = True
		sys.path.insert(benchmarkZero, str(proto))
		os.chdir(proto)
		protected = [Path(config['repository']).resolve(), (configurationDirectory / benchmarkSnapshotDirectory).resolve()]
		installWriteProtection(protected)
		import torch
		torch.set_num_threads(benchmarkThreadCount)
		torch.set_num_interop_threads(benchmarkThreadCount)
		defs = importlib.import_module(benchmarkCommonModule)
		validateSettings(defs, config)
		database = Path(defs.databaseFolder).resolve()
		protected.append(database)
		cache = configurationDirectory / benchmarkRuntimeCacheDirectory if mode == benchmarkPrepareMode else Path(config['output']) / benchmarkRuntimeCacheDirectory
		cache.mkdir(exist_ok=True)
		os.environ[benchmarkMatplotlibEnvironmentKey] = str(cache)
		if(mode == benchmarkPrepareMode):
			preparePlan(config, defs, configurationDirectory, database)
		elif(mode == benchmarkRunMode):
			runInference(config, defs, configurationDirectory, database, torch)
		else:
			raise ValueError(benchmarkErrorConfiguration + mode)
	return


def preparePlan(config, defs, directory, database):
	if(generalisedBenchmarkTool):
		configuredPrompt = Path(defs.inferencePromptFile).resolve()
		source = configuredPrompt
		if(defs.inferenceCopyTemplateDatasets and configuredPrompt == (database / defs.inferencePromptFileName).resolve()):
			source = (Path(defs.databaseFolderTemplate) / defs.inferencePromptFileName).resolve()
		if(not source.is_file()):
			raise ValueError(benchmarkErrorPrompt + str(source))
		data = source.read_bytes()
		lines = [line for line in data.decode(benchmarkEncoding).splitlines() if line.strip()]
		if(not lines):
			raise ValueError(benchmarkErrorPrompt + str(source))
		count = len(lines)
		if(config['max_sequences'] is not None):
			if(config['max_sequences'] > count):
				raise ValueError(benchmarkErrorPartial)
			count = config['max_sequences']
		(directory / benchmarkPromptFilename).write_bytes(data)
		settings = scalarSettings(defs)
		writeJSON(directory / benchmarkSettingsFilename, settings)
		plan = dict(configuration=config['configuration'], repository=config['repository'], configuration_directory=str(directory), database=str(database), configured_prompt=str(configuredPrompt), source_prompt=str(source), prompt_sha256=hashBytes(data), prompt_lines=len(lines), selected_prompt_lines=count, sampled=count != len(lines), max_sequences=config['max_sequences'], sequences_per_job=config['sequences_per_job'], runtime_overrides=benchmarkRuntimeOverrides, settings=settings)
		writeJSON(directory / benchmarkPlanFilename, plan)
	return


def runInference(config, defs, directory, database, torch):
	if(generalisedBenchmarkTool):
		plan = readJSON(directory / benchmarkPlanFilename)
		if(str(database) != plan['database'] or scalarSettings(defs) != plan['settings']):
			raise RuntimeError(benchmarkErrorChanged + str(directory))
		output = Path(config['output'])
		data = (directory / benchmarkPromptFilename).read_bytes()
		if(hashBytes(data) != plan['prompt_sha256']):
			raise RuntimeError(benchmarkErrorChanged + str(directory / benchmarkPromptFilename))
		if(config['sliced']):
			lines = [line for line in data.decode(benchmarkEncoding).splitlines() if line.strip()]
			data = (benchmarkNewline.join(lines[config['start']:config['stop']]) + benchmarkNewline).encode(benchmarkEncoding)
		prompt = output / benchmarkPromptFilename
		prompt.write_bytes(data)
		overrides = dict(benchmarkRuntimeOverrides, inferencePromptFile=str(prompt))
		for moduleName in (benchmarkCommonModule, benchmarkNLPDefsModule):
			module = sys.modules.get(moduleName)
			if(module is not None):
				for name, value in overrides.items():
					if(hasattr(module, name)):
						setattr(module, name, value)
		writeJSON(output / benchmarkSettingsFilename, dict(settings=scalarSettings(defs), runtime_overrides=overrides, python=sys.version, torch=torch.__version__, job=config))
		started = time.perf_counter()
		mainModule = importlib.import_module(benchmarkMainModule)
		prediction = importlib.import_module(benchmarkPredictionModule)
		nlp = importlib.import_module(benchmarkNLPModule)
		recorder = SequenceRecorder(prediction, nlp.processSequence, output, started)
		nlp.processSequence = recorder.processSequence
		try:
			mainModule.main()
		finally:
			recorder.close()
		matches = int(prediction.totalInferenceTop1PredictionMatches)
		tokens = int(prediction.totalInferenceTop1PredictionTokens)
		if(tokens <= benchmarkZero):
			raise RuntimeError(benchmarkErrorNoTokens)
		if(not benchmarkZero <= matches <= tokens or matches != recorder.matches or tokens != recorder.tokens):
			raise RuntimeError(benchmarkErrorResult + config['id'])
		result = dict(job_id=config['id'], job_sha256=hashBytes(Path(config['job_file']).read_bytes()), configuration=config['configuration'], database=str(database), prompt_sha256=plan['prompt_sha256'], selected_prompt_sha256=hashBytes(data), start=config['start'], stop=config['stop'], sampled=plan['sampled'], sequences=recorder.sequences, matches=matches, tokens=tokens, accuracy=matches/tokens, elapsed=time.perf_counter()-started, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		writeJSON(output / benchmarkResultFilename, result)
		print(json.dumps(result), flush=True)
	return


class SequenceRecorder:
	def __init__(self, prediction, original, output, started):
		if(generalisedBenchmarkTool):
			self.prediction = prediction
			self.original = original
			self.output = output
			self.started = started
			self.sequences = benchmarkZero
			self.matches = benchmarkZero
			self.tokens = benchmarkZero
			self.stream = (output / benchmarkSequencesFilename).open('w', encoding=benchmarkEncoding)
		return

	def processSequence(self, *args, **kwargs):
		result = None
		if(generalisedBenchmarkTool):
			beforeMatches = self.prediction.totalInferenceTop1PredictionMatches
			beforeTokens = self.prediction.totalInferenceTop1PredictionTokens
			result = self.original(*args, **kwargs)
			matches = int(self.prediction.totalInferenceTop1PredictionMatches - beforeMatches)
			tokens = int(self.prediction.totalInferenceTop1PredictionTokens - beforeTokens)
			if(not benchmarkZero <= matches <= tokens):
				raise RuntimeError(benchmarkErrorResult + str(self.output))
			self.sequences += benchmarkOne
			self.matches += matches
			self.tokens += tokens
			row = dict(sequence=int(args[benchmarkSequenceCountArgument]), matches=matches, tokens=tokens, elapsed=time.perf_counter()-self.started)
			self.stream.write(json.dumps(row) + benchmarkNewline)
			self.stream.flush()
			writeJSON(self.output / benchmarkProgressFilename, dict(sequences=self.sequences, matches=self.matches, tokens=self.tokens, elapsed=row['elapsed']))
		return result

	def close(self):
		if(generalisedBenchmarkTool):
			self.stream.close()
		return


def validateSettings(defs, config):
	if(generalisedBenchmarkTool):
		required = dict(benchmarkRequiredSettings)
		if(config['sequences_per_job'] > benchmarkZero or config['max_sequences'] is not None):
			required.update(benchmarkChunkRequiredSettings)
		for name, value in required.items():
			if(not hasattr(defs, name) or getattr(defs, name) != value):
				raise ValueError(benchmarkErrorConfiguration + name + '=' + repr(getattr(defs, name, None)) + '; required ' + repr(value))
		for name in benchmarkOptionalFalseSettings:
			if(hasattr(defs, name) and getattr(defs, name)):
				raise ValueError(benchmarkErrorConfiguration + name)
		if(not Path(defs.databaseFolder).is_dir() or not Path(defs.observedColumnsDir).is_dir()):
			raise ValueError(benchmarkErrorDatabase + defs.databaseFolder)
	return


def scalarSettings(defs):
	result = {}
	if(generalisedBenchmarkTool):
		result = {name:value for name,value in vars(defs).items() if not name.startswith('_') and (value is None or isinstance(value, (str, int, float, bool)))}
	return result


def installWriteProtection(protected):
	if(generalisedBenchmarkTool):
		def audit(event, arguments):
			if(generalisedBenchmarkTool):
				paths = []
				if(event == benchmarkAuditOpenEvent and isinstance(arguments[benchmarkZero], (str, bytes, os.PathLike)) and arguments[benchmarkAuditOpenFlagsIndex] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
					paths.append((arguments[benchmarkZero], None))
				elif(event in benchmarkAuditSinglePathEvents):
					paths.append((arguments[benchmarkZero], arguments[-benchmarkOne] if event in benchmarkAuditRemoveEvents else None))
				elif(event == benchmarkAuditMakeDirectoryEvent):
					paths.append((arguments[benchmarkZero], arguments[benchmarkAuditSourceDescriptorIndex]))
				elif(event in benchmarkAuditMultiplePathEvents):
					paths.extend(((arguments[benchmarkZero], arguments[benchmarkAuditSourceDescriptorIndex]), (arguments[benchmarkOne], arguments[benchmarkAuditTargetDescriptorIndex])))
				elif(event == benchmarkAuditSymlinkEvent):
					paths.append((arguments[benchmarkOne], arguments[benchmarkAuditSourceDescriptorIndex]))
				for path, descriptor in paths:
					if(isinstance(path, (str, bytes, os.PathLike))):
						resolved = Path(os.fsdecode(path))
						if(not resolved.is_absolute() and isinstance(descriptor, int) and descriptor >= benchmarkZero):
							resolved = Path(benchmarkProcFileDescriptors) / str(descriptor) / resolved
						resolved = resolved.resolve()
						if(any(resolved.is_relative_to(root) for root in protected)):
							raise RuntimeError(benchmarkErrorWrite + str(resolved))
			return
		sys.addaudithook(audit)
	return


if(__name__ == '__main__'):
	main()
