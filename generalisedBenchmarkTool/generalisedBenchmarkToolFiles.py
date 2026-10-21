"""Immutable source snapshots, line replacement and result serialization."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from generalisedBenchmarkTool_globalDefs import *


def prepareConfiguration(snapshot, configuration, destination):
	if(generalisedBenchmarkTool):
		shutil.copytree(snapshot, destination / benchmarkSnapshotDirectory)
		configurationBytes = configuration.read_bytes()
		(destination / benchmarkConfigurationFilename).write_bytes(configurationBytes)
		edits = parseEdits(configurationBytes.decode(benchmarkEncoding), snapshot)
		grouped = {}
		for edit in edits:
			grouped.setdefault(edit['file'], []).append(edit)
		diff = []
		for relative, replacements in grouped.items():
			path = destination / benchmarkSnapshotDirectory / relative
			original = path.read_bytes().decode(benchmarkEncoding)
			lines = original.splitlines(keepends=True)
			for edit in replacements:
				index = edit['line'] - benchmarkOne
				old = lines[index]
				ending = old[len(old.rstrip('\r\n')):]
				edit['original'] = old.rstrip('\r\n')
				lines[index] = edit['replacement'] + ending
			updated = ''.join(lines)
			path.write_bytes(updated.encode(benchmarkEncoding))
			diff.extend(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True), fromfile=relative, tofile=relative))
		validatePython(destination / benchmarkSnapshotDirectory)
		(destination / benchmarkDiffFilename).write_text(''.join(diff), encoding=benchmarkEncoding)
		writeJSON(destination / benchmarkEditsFilename, edits)
	return


def parseEdits(text, snapshot):
	edits = []
	if(generalisedBenchmarkTool):
		seen = set()
		for entry in text.splitlines():
			if(entry.strip() and not entry.lstrip().startswith(benchmarkCommentPrefix)):
				match = re.fullmatch(benchmarkEditPattern, entry)
				if(match is None):
					raise ValueError(benchmarkErrorConfig + entry)
				relative, number, replacement = match.groups()
				path = validateSourcePath(snapshot, relative)
				if(path.suffix == benchmarkRequirementExtension):
					raise ValueError(benchmarkErrorPath + relative)
				line = int(number)
				key = (relative, line)
				if(key in seen):
					raise ValueError(benchmarkErrorDuplicate + entry)
				if(line > len(path.read_bytes().decode(benchmarkEncoding).splitlines())):
					raise ValueError(benchmarkErrorRange + entry)
				seen.add(key)
				edits.append(dict(file=relative, line=line, replacement=decodeIndentation(replacement)))
	return edits


def decodeIndentation(replacement):
	result = replacement
	if(generalisedBenchmarkTool):
		indentation = []
		position = benchmarkZero
		while(position < len(replacement)):
			if(replacement.startswith(benchmarkLiteralTab, position)):
				indentation.append(benchmarkTab)
				position += len(benchmarkLiteralTab)
			elif(replacement[position] in (benchmarkConfigLiteralSpace, benchmarkTab)):
				indentation.append(replacement[position])
				position += benchmarkOne
			else:
				break
		result = ''.join(indentation) + replacement[position:]
	return result


def snapshotRepository(repository, destination):
	if(generalisedBenchmarkTool):
		listing = subprocess.run([*benchmarkGitSnapshotCommand, str(repository), *benchmarkGitSnapshotArguments], check=True, capture_output=True).stdout
		destination.mkdir(parents=True)
		for relative in sorted(set(os.fsdecode(listing).split(benchmarkNul))):
			if(relative and not benchmarkForbiddenParts.intersection(Path(relative).parts)):
				candidate = repository / relative
				if(candidate.exists() or candidate.is_symlink()):
					path = validateSourcePath(repository, relative)
					target = destination / relative
					target.parent.mkdir(parents=True, exist_ok=True)
					shutil.copy2(path, target)
		validatePython(destination)
	return


def validateSourcePath(root, relative):
	path = root / relative
	if(generalisedBenchmarkTool):
		parts = Path(relative).parts
		if(str(Path(relative)) != relative or Path(relative).is_absolute() or benchmarkParentPathComponent in parts or benchmarkForbiddenParts.intersection(parts) or not parts or not path.is_file() or not path.resolve().is_relative_to(root.resolve())):
			raise ValueError(benchmarkErrorPath + relative)
		cursor = root
		for part in parts:
			cursor /= part
			if(cursor.is_symlink()):
				raise ValueError(benchmarkErrorPath + relative)
	return path


def validatePython(root):
	if(generalisedBenchmarkTool):
		for path in sorted(root.rglob('*' + benchmarkPythonExtension)):
			try:
				compile(path.read_bytes(), str(path), 'exec')
			except (SyntaxError, ValueError) as error:
				raise ValueError(benchmarkErrorSyntax + str(path)) from error
	return


def fileHashes(root):
	result = {}
	if(generalisedBenchmarkTool):
		for path in sorted(root.rglob('*')):
			if(path.is_symlink()):
				raise ValueError(benchmarkErrorPath + str(path))
			if(path.is_file()):
				result[str(path.relative_to(root))] = hashBytes(path.read_bytes())
	return result


def verifyHashes(root, expected):
	if(generalisedBenchmarkTool):
		if(fileHashes(root) != expected):
			raise RuntimeError(benchmarkErrorChanged + str(root))
	return


def hashBytes(data):
	result = None
	if(generalisedBenchmarkTool):
		result = hashlib.sha256(data).hexdigest()
	return result


def writeJSON(path, value):
	if(generalisedBenchmarkTool):
		path = Path(path)
		temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + benchmarkTemporaryExtension)
		temporary.write_text(json.dumps(value, indent=benchmarkJSONIndent, allow_nan=False) + benchmarkNewline, encoding=benchmarkEncoding)
		os.replace(temporary, path)
	return


def readJSON(path):
	result = None
	if(generalisedBenchmarkTool):
		result = json.loads(Path(path).read_text(encoding=benchmarkEncoding))
	return result
