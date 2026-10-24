"""Subprocess integration tests using the real native scheduler with a small model fixture."""
import ast
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from generalisedBenchmarkTool_globalDefs import *
from generalisedBenchmarkToolFiles import applyExecutionSettings, fileHashes, parseEdits, prepareConfiguration, readJSON
from generalisedBenchmarkToolTest_globalDefs import *


class BenchmarkTests(unittest.TestCase):
	def setUp(self):
		if(generalisedBenchmarkTool):
			self.temporary = tempfile.TemporaryDirectory()
			self.root = Path(self.temporary.name)
			self.repository = self.root / 'repository'
			proto = self.repository / benchmarkProtoDirectory
			proto.mkdir(parents=True)
			self.configurations = self.root / benchmarkConfigurationsDirectory
			self.configurations.mkdir()
			self.runs = self.root / 'runs'
			self.database = self.root / 'database'
			(self.database / 'observedColumns').mkdir(parents=True)
			self.marker = self.root / 'fail'
			self.writeMarker = self.root / 'write'
			settings = dict(matchesPerSequence=benchmarkTestFirstMatches, **benchmarkTestNativeRequired, databaseFolder=str(self.database), observedColumnsDir=str(self.database/'observedColumns'), conceptColumnsDictFile=str(self.database/'columns'), conceptFeaturesDictFile=str(self.database/'features'), inferencePromptFile=str(self.database/'inference.txt'), inferencePromptFileName='inference.txt', databaseFolderTemplate='../databaseTemplate', failureMarker=str(self.marker), writeMarker=str(self.writeMarker), failingMatches=benchmarkTestSecondMatches, testDelay=benchmarkTestSequenceDelay, testTokens=benchmarkTestTokensPerSequence, numSeedTokensInference=benchmarkTestSeedTokens, inferenceChunksSequenceCount=benchmarkTestChunkSize, inferenceWorkersCount=benchmarkTestWorkerCount)
			for name in ('columns', 'features'):
				(self.database / name).touch()
			nativeDefs = (benchmarkDirectory.parent / benchmarkProtoDirectory / (benchmarkCommonModule + benchmarkPythonExtension)).read_text()
			constants = next(node for node in ast.parse(nativeDefs).body if isinstance(node, ast.If) and ast.unparse(node.test) == 'optimisationInferenceChunks or optimisationInferenceWorkers')
			text = benchmarkNewline.join(name + ' = ' + repr(value) for name,value in settings.items()) + benchmarkNewline
			text += benchmarkNewline.join(nativeDefs.splitlines()[constants.lineno-benchmarkOne:constants.end_lineno]) + benchmarkNewline
			(proto / (benchmarkCommonModule + benchmarkPythonExtension)).write_text(text)
			for name in (benchmarkChunksModule, benchmarkNativeWorkerModule):
				shutil.copy2(benchmarkDirectory.parent / benchmarkProtoDirectory / (name + benchmarkPythonExtension), proto / (name + benchmarkPythonExtension))
			fixtures = {'extra.py':"value = 'original'\n", 'torch.py':benchmarkTestTorch, benchmarkTestPredictionModule+benchmarkPythonExtension:benchmarkTestPrediction, benchmarkMainModule+benchmarkPythonExtension:benchmarkTestMain, benchmarkTestNLPModule+benchmarkPythonExtension:benchmarkTestNLP, benchmarkTestDebugModule+benchmarkPythonExtension:benchmarkTestDebug}
			for name, text in fixtures.items():
				(proto / name).write_text(text)
			(self.repository / 'databaseTemplate').mkdir()
			(self.repository / 'databaseTemplate/inference.txt').write_text(benchmarkTestInput)
			(self.database / 'inference.txt').write_text(benchmarkTestStalePrompt)
			(self.configurations / 'a.txt').write_text('# Baseline\n')
			(self.configurations / 'b.txt').write_text(benchmarkTestConfigurationText)
			subprocess.run(['git', 'init', '-q', str(self.repository)], check=True)
			self.originalHashes = fileHashes(proto)
		return

	def tearDown(self):
		if(generalisedBenchmarkTool):
			self.temporary.cleanup()
		return

	def testEdits(self):
		if(generalisedBenchmarkTool):
			fixture = self.repository / 'example.py'
			fixture.write_bytes(b'if True:\r\n\tvalue = "old"\r\n\tother = 2')
			config = self.configurations / 'literal.txt'
			config.write_text('example.py:line2:\\tvalue = r"a\\tb:colon"\nexample.py:line3:\tother = 4\n')
			destination = self.root / 'patched'
			prepareConfiguration(self.repository, config, destination)
			self.assertEqual((destination / benchmarkSnapshotDirectory / 'example.py').read_bytes(), b'if True:\r\n\tvalue = r"a\\tb:colon"\r\n\tother = 4')
			for invalid in ('example.py:line0:x', 'example.py:line4:x', 'example.py:line1:x\nexample.py:line1:y', '../outside:line1:x', '/etc/passwd:line1:x', './example.py:line1:x'):
				with self.subTest(invalid=invalid), self.assertRaises(ValueError):
					parseEdits(invalid, self.repository)
			(self.repository / 'escape.py').symlink_to(fixture)
			with self.assertRaises(ValueError):
				parseEdits('escape.py:line1:x', self.repository)
		return

	def testNativeDelegationAndCompletedResume(self):
		if(generalisedBenchmarkTool):
			self.execute('-numWorkers', str(benchmarkTestWorkerCount), '-sequencesPerJob', str(benchmarkTestChunkSize))
			run = self.singleRun()
			summary = readJSON(run / benchmarkSummaryFilename)
			self.assertEqual(summary['status'], benchmarkStatusComplete)
			self.assertEqual(summary['completed_configurations'], benchmarkTestConfigurationCount)
			self.assertEqual([r['matches'] for r in summary['configurations']], [benchmarkTestFirstMatches*benchmarkTestSequenceCount, benchmarkTestSecondMatches*benchmarkTestSequenceCount])
			self.assertEqual([r['tokens'] for r in summary['configurations']], [benchmarkTestTokensPerSequence*benchmarkTestSequenceCount]*benchmarkTestConfigurationCount)
			self.assertEqual(fileHashes(self.repository / benchmarkProtoDirectory), self.originalHashes)
			self.assertEqual((self.database / 'inference.txt').read_text(), benchmarkTestStalePrompt)
			for row in summary['configurations']:
				native = Path(row['native_run'])
				manifest = readJSON(native / benchmarkManifestFilename)
				self.assertEqual(manifest['settings'][benchmarkWorkerCountSetting], benchmarkTestWorkerCount)
				self.assertEqual(manifest['settings'][benchmarkChunkSizeSetting], benchmarkTestChunkSize)
				self.assertEqual(manifest['selected_sequences'], benchmarkTestSequenceCount)
				self.assertEqual(len(manifest['jobs']), benchmarkTestConfigurationCount)
			first = run / benchmarkOutputDirectory / 'a' / benchmarkResultFilename
			mtime = first.stat().st_mtime_ns
			self.execute('-resume', str(run), newRun=False)
			self.assertEqual(first.stat().st_mtime_ns, mtime)
			(run / benchmarkConfigurationsDirectory / 'a' / benchmarkSnapshotDirectory / benchmarkProtoDirectory / 'extra.py').write_text('value = 100\n')
			result = self.execute('-resume', str(run), newRun=False, success=False)
			self.assertIn(benchmarkErrorChanged, result.stdout)
		return

	def testRepositoryLocalRunsAndSnapshotExclusion(self):
		if(generalisedBenchmarkTool):
			tool = self.repository / benchmarkDirectory.name
			tool.mkdir()
			for filename in (*benchmarkEngineFilenames, benchmarkTestIgnoreFilename):
				shutil.copy2(benchmarkDirectory / filename, tool / filename)
			runs = tool / benchmarkRunsDirectoryName
			historical = runs / benchmarkTestHistoricalDirectory
			historical.mkdir(parents=True)
			invalid = historical / benchmarkTestHistoricalFilename
			invalid.write_text(benchmarkTestInvalidSource)
			subprocess.run(['git', '-C', str(self.repository), 'add', '-f', str(invalid)], check=True)
			for attempt in range(benchmarkTestRepeatCount):
				result = subprocess.run([sys.executable, str(tool / benchmarkDriverFilename), '-configurations', str(self.configurations), '-numConfigurations', str(benchmarkOne)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=benchmarkTestDeadline, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
				self.assertEqual(result.returncode, benchmarkExitSuccess, result.stdout)
			completed = [path for path in runs.iterdir() if path != historical]
			self.assertEqual(len(completed), benchmarkTestRepeatCount)
			for run in completed:
				self.assertEqual(readJSON(run / benchmarkSummaryFilename)['status'], benchmarkStatusComplete)
				self.assertFalse((run / benchmarkSnapshotDirectory / benchmarkRepositoryRunsPath).exists())
				self.execute('-resume', str(run), newRun=False)
				ignored = subprocess.run(['git', '-C', str(self.repository), 'check-ignore', str(run / benchmarkSummaryFilename)], capture_output=True, text=True)
				self.assertEqual(ignored.returncode, benchmarkExitSuccess, ignored.stderr)
			self.assertEqual(fileHashes(self.repository / benchmarkProtoDirectory), self.originalHashes)
		return

	def testInvalidOutputLocations(self):
		if(generalisedBenchmarkTool):
			for output in (self.repository, self.root, self.configurations, self.configurations / benchmarkRunsDirectoryName, self.repository / benchmarkProtoDirectory / benchmarkRunsDirectoryName, self.repository / '.git' / benchmarkRunsDirectoryName):
				with self.subTest(output=output):
					result = self.execute('-runsDirectory', str(output), '-prepareOnly', success=False)
					self.assertIn(benchmarkErrorLocation, result.stdout)
		return

	def testConfigurationFailureThenResume(self):
		if(generalisedBenchmarkTool):
			self.marker.touch()
			self.execute(success=False)
			run = self.singleRun()
			summary = readJSON(run / benchmarkSummaryFilename)
			self.assertEqual(summary['status'], benchmarkStatusFailed)
			self.assertEqual(summary['completed_configurations'], benchmarkOne)
			self.assertIsNone(summary['configurations'][benchmarkOne]['accuracy_percent'])
			first = run / benchmarkOutputDirectory / 'a' / benchmarkResultFilename
			mtime = first.stat().st_mtime_ns
			self.marker.unlink()
			self.execute('-resume', str(run), newRun=False)
			self.assertEqual(first.stat().st_mtime_ns, mtime)
			self.assertEqual(readJSON(run / benchmarkSummaryFilename)['status'], benchmarkStatusComplete)
			self.assertEqual(len(list((run / benchmarkHistoryDirectory / 'b').iterdir())), benchmarkOne)
		return

	def testDatabaseWritesFail(self):
		if(generalisedBenchmarkTool):
			self.writeMarker.touch()
			self.execute('-numConfigurations', str(benchmarkOne), success=False)
			self.assertFalse((self.database / 'must-not-write.txt').exists())
			run = self.singleRun()
			self.assertEqual(readJSON(run / benchmarkSummaryFilename)['status'], benchmarkStatusFailed)
			logs = list((run / benchmarkOutputDirectory / 'a' / benchmarkNativeDirectory).rglob(benchmarkLogFilename))
			self.assertTrue(any('attempted to modify a protected input' in p.read_text() for p in logs))
		return

	def testInvalidEditStopsBeforeInference(self):
		if(generalisedBenchmarkTool):
			(self.configurations / 'b.txt').write_text('proto/extra.py:line1:if broken syntax\n')
			result = self.execute(success=False)
			self.assertIn(benchmarkErrorSyntax, result.stdout)
			self.assertFalse(list(self.runs.rglob(benchmarkResultFilename)))
			self.assertEqual(fileHashes(self.repository / benchmarkProtoDirectory), self.originalHashes)
		return

	def testInterruptedNativeWorkersAndLock(self):
		if(generalisedBenchmarkTool):
			self.setFixtureSetting('testDelay', benchmarkTestSequenceCount)
			self.execute('-prepareOnly', '-numConfigurations', str(benchmarkOne))
			run = self.singleRun()
			process = subprocess.Popen(self.command('-resume', str(run)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
			try:
				self.waitFor(lambda: bool(self.activeNativeWorkers(run)))
				pids = self.activeNativeWorkers(run)
				blocked = self.execute('-resume', str(run), newRun=False, success=False)
				self.assertIn(benchmarkErrorLock, blocked.stdout)
				process.send_signal(signal.SIGTERM)
				output, unused = process.communicate(timeout=benchmarkTestDeadline)
				self.assertNotEqual(process.returncode, benchmarkExitSuccess, output)
				for pid in pids:
					with self.assertRaises(ProcessLookupError):
						os.kill(pid, benchmarkZero)
			finally:
				if(process.poll() is None):
					process.terminate()
					process.wait(timeout=benchmarkTestDeadline)
			self.execute('-resume', str(run), newRun=False)
			self.assertEqual(readJSON(run / benchmarkSummaryFilename)['status'], benchmarkStatusComplete)
		return

	def testSamplingAndInheritedOptions(self):
		if(generalisedBenchmarkTool):
			self.execute('-maxSequences', str(benchmarkOne))
			summary = readJSON(self.singleRun() / benchmarkSummaryFilename)
			for row in summary['configurations']:
				self.assertTrue(row['sampled'])
				self.assertEqual(row['sequences'], benchmarkOne)
				self.assertEqual(row['tokens'], benchmarkTestTokensPerSequence)
		return

	def testSerialNativeChunks(self):
		if(generalisedBenchmarkTool):
			self.setFixtureSetting(benchmarkWorkersSetting, False)
			self.execute('-numConfigurations', str(benchmarkOne))
			self.assertEqual(readJSON(self.singleRun() / benchmarkSummaryFilename)['status'], benchmarkStatusComplete)
		return

	def testInvalidNativeAndLegacyOptions(self):
		if(generalisedBenchmarkTool):
			for arguments in (('-numWorkers','0'), ('-sequencesPerJob','0')):
				with self.subTest(arguments=arguments):
					self.assertIn(benchmarkErrorPositive, self.execute(*arguments, success=False).stdout)
			self.setFixtureSetting('executionMode', 'train')
			self.execute(success=False)
			self.assertFalse(list(self.runs.rglob(benchmarkResultFilename)))
		return

	def testConflictingAssignmentsFail(self):
		if(generalisedBenchmarkTool):
			path = self.repository / benchmarkProtoDirectory / (benchmarkCommonModule + benchmarkPythonExtension)
			path.write_text(path.read_text() + 'inferenceWorkersCount = 99\n')
			with self.assertRaisesRegex(ValueError, benchmarkErrorSetting):
				applyExecutionSettings(self.repository, {benchmarkWorkerCountSetting:benchmarkTestWorkerCount})
		return

	def testCorruptResultFailsResume(self):
		if(generalisedBenchmarkTool):
			self.execute('-numConfigurations', str(benchmarkOne))
			run = self.singleRun()
			row = readJSON(run / benchmarkSummaryFilename)['configurations'][benchmarkZero]
			path = Path(row['native_run']) / benchmarkSummaryFilename
			path.write_text('{}')
			self.assertIn(benchmarkErrorChanged, self.execute('-resume', str(run), newRun=False, success=False).stdout)
		return

	def setFixtureSetting(self, name, value):
		if(generalisedBenchmarkTool):
			applyExecutionSettings(self.repository, {name:value})
		return

	def activeNativeWorkers(self, run):
		if(generalisedBenchmarkTool):
			result = []
			for p in (run / benchmarkOutputDirectory).glob('*/native/*/summary.json'):
				result.extend(readJSON(p)['active_workers'].values())
		return result

	def command(self, *arguments):
		if(generalisedBenchmarkTool):
			result = [sys.executable, str(benchmarkDirectory / benchmarkDriverFilename), *arguments]
		return result

	def execute(self, *arguments, newRun=True, success=True):
		if(generalisedBenchmarkTool):
			common = ['-repository', str(self.repository), '-configurations', str(self.configurations), '-runsDirectory', str(self.runs)] if newRun else []
			result = subprocess.run(self.command(*common, *arguments), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=benchmarkTestDeadline, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
			if(success):
				self.assertEqual(result.returncode, benchmarkExitSuccess, result.stdout)
			else:
				self.assertNotEqual(result.returncode, benchmarkExitSuccess, result.stdout)
		return result

	def singleRun(self):
		if(generalisedBenchmarkTool):
			entries = list(self.runs.iterdir())
			self.assertEqual(len(entries), benchmarkOne)
			result = entries[benchmarkZero]
		return result

	def waitFor(self, predicate):
		if(generalisedBenchmarkTool):
			deadline = time.monotonic() + benchmarkTestDeadline
			while(not predicate() and time.monotonic() < deadline):
				time.sleep(benchmarkTestPoll)
			self.assertTrue(predicate())
		return


if(__name__ == '__main__'):
	unittest.main(verbosity=benchmarkJSONIndent)
