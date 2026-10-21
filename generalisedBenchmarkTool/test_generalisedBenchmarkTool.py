"""Small unit and subprocess integration tests; no trained database is modified."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from generalisedBenchmarkTool_globalDefs import *
from generalisedBenchmarkToolFiles import fileHashes, parseEdits, prepareConfiguration, readJSON

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
			self.runs = self.root / benchmarkRunsDirectory
			self.database = self.root / 'database'
			(self.database / 'observedColumns').mkdir(parents=True)
			self.marker = self.root / 'fail'
			self.writeMarker = self.root / 'write'
			settings = dict(matchesPerSequence=benchmarkTestFirstMatches, **benchmarkRequiredSettings, **benchmarkChunkRequiredSettings, databaseFolder=str(self.database), observedColumnsDir=str(self.database / 'observedColumns'), inferencePromptFile=str(self.database / 'inference.txt'), inferencePromptFileName='inference.txt', databaseFolderTemplate='../databaseTemplate', inferenceCopyTemplateDatasets=True, failureMarker=str(self.marker), writeMarker=str(self.writeMarker), testDelay=benchmarkTestSequenceDelay, testTokens=benchmarkTestTokensPerSequence)
			(proto / (benchmarkCommonModule + benchmarkPythonExtension)).write_text(benchmarkNewline.join(name + ' = ' + repr(value) for name,value in settings.items()) + benchmarkNewline)
			(proto / 'extra.py').write_text("value = 'original'\n")
			(proto / 'torch.py').write_text(benchmarkTestTorch)
			(proto / (benchmarkPredictionModule + benchmarkPythonExtension)).write_text(benchmarkTestPrediction)
			(proto / (benchmarkMainModule + benchmarkPythonExtension)).write_text(benchmarkTestMain)
			(proto / (benchmarkNLPModule + benchmarkPythonExtension)).write_text(benchmarkTestNLP)
			(self.repository / 'databaseTemplate').mkdir()
			(self.repository / 'databaseTemplate/inference.txt').write_text('first\nsecond\nthird\n')
			(self.database / 'inference.txt').write_text('stale database prompt\n')
			(self.configurations / 'a.txt').write_text('# Unmodified baseline\n')
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
			self.assertEqual(fixture.read_bytes(), b'if True:\r\n\tvalue = "old"\r\n\tother = 2')
			for invalid in ('example.py:line0:x', 'example.py:line4:x', 'example.py:line1:x\nexample.py:line1:y', '../outside:line1:x', '/etc/passwd:line1:x', './example.py:line1:x', 'proto//extra.py:line1:x'):
				with self.subTest(invalid=invalid), self.assertRaises(ValueError):
					parseEdits(invalid, self.repository)
			(self.repository / 'escape.py').symlink_to(fixture)
			with self.assertRaises(ValueError):
				parseEdits('escape.py:line1:x', self.repository)
		return

	def testParallelFullResultsAndResume(self):
		if(generalisedBenchmarkTool):
			self.execute('-prepareOnly')
			run = self.singleRun()
			process = subprocess.Popen(self.command('-numWorkers', '2', '-resume', str(run)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
			try:
				self.waitFor(lambda: len(readJSON(run / benchmarkSummaryFilename)['active_jobs']) == benchmarkJSONIndent)
				output, unused = process.communicate(timeout=benchmarkTestDeadline)
				self.assertEqual(process.returncode, benchmarkExitSuccess, output)
			finally:
				if(process.poll() is None):
					process.terminate()
					process.wait()
			summary = readJSON(run / benchmarkSummaryFilename)
			self.assertEqual(summary['status'], benchmarkStatusComplete)
			self.assertEqual([row['matches'] for row in summary['configurations']], [benchmarkTestFirstMatches*3, benchmarkTestSecondMatches*3])
			self.assertEqual([row['tokens'] for row in summary['configurations']], [benchmarkTestTokensPerSequence*3]*2)
			self.assertEqual(fileHashes(self.repository / benchmarkProtoDirectory), self.originalHashes)
			self.assertEqual((self.database / 'inference.txt').read_text(), 'stale database prompt\n')
			resultFiles = list((run / benchmarkJobsDirectory).glob('*/output/result.json'))
			before = [path.stat().st_mtime_ns for path in resultFiles]
			self.execute('-resume', str(run), newRun=False)
			self.assertEqual(before, [path.stat().st_mtime_ns for path in resultFiles])
			(run / benchmarkConfigurationsDirectory / 'a' / benchmarkSnapshotDirectory / benchmarkProtoDirectory / 'extra.py').write_text('value = 100\n')
			result = self.execute('-resume', str(run), newRun=False, success=False)
			self.assertIn(benchmarkErrorChanged, result.stdout)
		return

	def testChunkFailureThenResume(self):
		if(generalisedBenchmarkTool):
			self.marker.touch()
			self.execute('-numConfigurations', '1', '-sequencesPerJob', '1', success=False)
			run = self.singleRun()
			summary = readJSON(run / benchmarkSummaryFilename)
			self.assertEqual(summary['status'], benchmarkStatusFailed)
			self.assertEqual(summary['completed_jobs'], benchmarkOne)
			first = run / benchmarkJobsDirectory / 'job000001' / benchmarkOutputDirectory / benchmarkResultFilename
			mtime = first.stat().st_mtime_ns
			self.marker.unlink()
			self.execute('-resume', str(run), newRun=False)
			self.assertEqual(first.stat().st_mtime_ns, mtime)
			row = readJSON(run / benchmarkSummaryFilename)['configurations'][benchmarkZero]
			self.assertEqual(row['matches'], benchmarkTestFirstMatches*3)
			self.assertEqual(row['tokens'], benchmarkTestTokensPerSequence*3)
			self.assertEqual(len(list((run / benchmarkJobsDirectory / 'job000002' / benchmarkHistoryDirectory).iterdir())), benchmarkOne)
		return

	def testDatabaseWritesFail(self):
		if(generalisedBenchmarkTool):
			self.writeMarker.touch()
			self.execute('-numConfigurations', '1', success=False)
			run = self.singleRun()
			log = run / benchmarkJobsDirectory / 'job000001' / benchmarkOutputDirectory / benchmarkLogFilename
			self.assertIn(benchmarkErrorWrite, log.read_text())
			self.assertFalse((self.database / 'must-not-write.txt').exists())
		return

	def testInvalidEditStopsBeforeInference(self):
		if(generalisedBenchmarkTool):
			(self.configurations / 'b.txt').write_text('proto/extra.py:line1:if broken syntax\n')
			result = self.execute(success=False)
			self.assertIn(benchmarkErrorSyntax, result.stdout)
			self.assertFalse(list(self.runs.rglob(benchmarkResultFilename)))
			self.assertEqual(fileHashes(self.repository / benchmarkProtoDirectory), self.originalHashes)
		return

	def testInterruptedWorkersAndLock(self):
		if(generalisedBenchmarkTool):
			self.execute('-prepareOnly', '-numConfigurations', '1', '-sequencesPerJob', '1')
			run = self.singleRun()
			process = subprocess.Popen(self.command('-numWorkers', '1', '-resume', str(run)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
			try:
				self.waitFor(lambda: bool(readJSON(run / benchmarkSummaryFilename)['active_jobs']))
				blocked = self.execute('-resume', str(run), newRun=False, success=False)
				self.assertIn(benchmarkErrorLock, blocked.stdout)
				process.send_signal(signal.SIGTERM)
				process.communicate(timeout=benchmarkTestDeadline)
				self.assertNotEqual(process.returncode, benchmarkExitSuccess)
			finally:
				if(process.poll() is None):
					process.terminate()
					process.wait()
			self.execute('-resume', str(run), newRun=False)
			self.assertEqual(readJSON(run / benchmarkSummaryFilename)['status'], benchmarkStatusComplete)
		return

	def command(self, *arguments):
		result = None
		if(generalisedBenchmarkTool):
			result = [sys.executable, str(benchmarkDirectory / benchmarkDriverFilename), *arguments]
		return result

	def execute(self, *arguments, newRun=True, success=True):
		result = None
		if(generalisedBenchmarkTool):
			common = ['-repository', str(self.repository), '-configurations', str(self.configurations), '-runsDirectory', str(self.runs), '-python', sys.executable] if newRun else []
			result = subprocess.run(self.command('-numWorkers', '1', *common, *arguments), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=benchmarkTestDeadline)
			if(success):
				self.assertEqual(result.returncode, benchmarkExitSuccess, result.stdout)
			else:
				self.assertNotEqual(result.returncode, benchmarkExitSuccess, result.stdout)
		return result

	def singleRun(self):
		result = None
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
