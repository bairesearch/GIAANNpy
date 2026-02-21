"""GIAANNproto_sequencePOS.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto sequence POS

# EVER-POS (all spaCy POS types) - FULL, CORRECT, FAST SOLUTION
# =============================================================
# - Exact "ever classifiable" semantics
# - WordNet + NLTK tagged corpora (no spaCy tagger usage)
# - Precomputed dictionaries
# - Safe caching (no unhashable arguments)
# - Save / load supported
# - Drop-in runnable

"""

import gzip
import json
import os
import pickle
import re
import string
import time
from functools import lru_cache
from typing import Dict, Set, Any, Optional
import nltk

from GIAANNproto_globalDefs import *


try:
	nltk.data.find('corpora/wordnet')
except LookupError:
	nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import treebank
from nltk.corpus import conll2000


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

SPACY_POS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")

WORDNET_POS_MAP = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
PTB_TAG_TO_SPACY_POS = {"CC": ("CCONJ",), "CD": ("NUM",), "DT": ("DET",), "EX": ("PRON",), "FW": ("X",), "IN": ("ADP", "SCONJ"), "JJ": ("ADJ",), "JJR": ("ADJ",), "JJS": ("ADJ",), "LS": ("X",), "MD": ("AUX",), "NN": ("NOUN",), "NNS": ("NOUN",), "NNP": ("PROPN",), "NNPS": ("PROPN",), "PDT": ("DET",), "POS": ("PART",), "PRP": ("PRON",), "PRP$": ("PRON",), "RB": ("ADV",), "RBR": ("ADV",), "RBS": ("ADV",), "RP": ("PART",), "SYM": ("SYM",), "TO": ("PART", "ADP"), "UH": ("INTJ",), "VB": ("VERB",), "VBD": ("VERB",), "VBG": ("VERB",), "VBN": ("VERB",), "VBP": ("VERB",), "VBZ": ("VERB",), "WDT": ("DET",), "WP": ("PRON",), "WP$": ("PRON",), "WRB": ("ADV",), "#": ("SYM",), "$": ("SYM",), ".": ("PUNCT",), ",": ("PUNCT",), ":": ("PUNCT",), "``": ("PUNCT",), "''": ("PUNCT",), "-LRB-": ("PUNCT",), "-RRB-": ("PUNCT",), "HYPH": ("PUNCT",), "NFP": ("PUNCT",), "ADD": ("X",), "AFX": ("ADJ",), "GW": ("X",), "XX": ("X",)}
POS_CORPORA = ("treebank", "conll2000")
CORPUS_LOOKUP_PATHS = {"treebank": "corpora/treebank", "conll2000": "corpora/conll2000"}
AUX_WORDS = {"am", "are", "is", "was", "were", "be", "being", "been", "do", "does", "did", "doing", "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought", "need", "dare"}
SYMBOL_CHARS = "$%&*+=<>@^~|"


# ------------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------------

def normalizeWord(word: str) -> str:
	return word.lower().replace(" ", "_")

def initPosDicts() -> Dict[str, Set[str]]:
	result = {p: set() for p in SPACY_POS}
	return result

def ensurePosDictsComplete(posDicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
	result = posDicts
	for posType in SPACY_POS:
		if posType not in result:
			result[posType] = set()
	return result

def ensureNltkCorpus(corpusName: str) -> None:
	lookupPath = CORPUS_LOOKUP_PATHS.get(corpusName)
	found = False
	if lookupPath is None:
		raise RuntimeError(f"ensureNltkCorpus error: missing lookup path for corpus '{corpusName}'")
	try:
		nltk.data.find(lookupPath)
		found = True
	except LookupError:
		nltk.download(corpusName)
		try:
			nltk.data.find(lookupPath)
			found = True
		except LookupError:
			found = False
	if not found:
		raise RuntimeError(f"ensureNltkCorpus error: corpus '{corpusName}' not available after download")

def getTaggedWordsFromCorpus(corpusName: str):
	result = []
	found = False
	if corpusName == "treebank":
		result = treebank.tagged_words()
		found = True
	elif corpusName == "conll2000":
		result = conll2000.tagged_words()
		found = True
	if not found:
		raise RuntimeError(f"getTaggedWordsFromCorpus error: unsupported corpus '{corpusName}'")
	return result

def addTaggedWordsToPosDicts(posDicts: Dict[str, Set[str]], taggedWords) -> Dict[str, Set[str]]:
	result = posDicts
	for word, tag in taggedWords:
		posList = PTB_TAG_TO_SPACY_POS.get(tag)
		if posList is None:
			continue
		normalizedWord = normalizeWord(word)
		for posType in posList:
			if posType in result:
				result[posType].add(normalizedWord)
	return result

def addWordNetPosDicts(posDicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
	result = posDicts
	lemmaPosMap = wn._lemma_pos_offset_map
	for lemma, posOffsets in lemmaPosMap.items():
		normalizedLemma = normalizeWord(lemma)
		for posType, wnPos in WORDNET_POS_MAP.items():
			if wnPos in posOffsets:
				result[posType].add(normalizedLemma)
	return result

def addAuxWordsToPosDicts(posDicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
	result = posDicts
	for word in AUX_WORDS:
		result["AUX"].add(word)
	return result

def addManualPosOverrides(posDicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
	result = posDicts
	overrides = {"ADP": set(("towards", "afterwards",)),}
	for posType, words in overrides.items():
		if(posType in result):
			result[posType].update(words)
	return result

def isNumericWord(word: str) -> bool:
	result = False
	if word:
		if re.fullmatch(r"[0-9]+([.,][0-9]+)*", word):
			result = True
	return result

def isPunctWord(word: str) -> bool:
	result = False
	if word:
		allPunct = True
		for ch in word:
			if ch not in string.punctuation:
				allPunct = False
		result = allPunct
	return result

def isSymbolWord(word: str) -> bool:
	result = False
	if word:
		hasSymbol = False
		for ch in word:
			if ch in SYMBOL_CHARS:
				hasSymbol = True
		result = hasSymbol
	return result

def posDictsMissingTypes(posDicts: Dict[str, Set[str]]) -> bool:
	result = False
	for posType in SPACY_POS:
		if posType not in posDicts:
			result = True
	return result


# ------------------------------------------------------------------
# BUILD (ONE-TIME)
# ------------------------------------------------------------------

def buildEverPosDicts() -> Dict[str, Set[str]]:
	"""
	Build EVER-POS dictionaries using WordNet and tagged corpora.
	"""
	posDicts = initPosDicts()
	posDicts = addWordNetPosDicts(posDicts)
	posDicts = addAuxWordsToPosDicts(posDicts)
	for corpusName in POS_CORPORA:
		ensureNltkCorpus(corpusName)
		taggedWords = getTaggedWordsFromCorpus(corpusName)
		posDicts = addTaggedWordsToPosDicts(posDicts, taggedWords)
	posDicts = ensurePosDictsComplete(posDicts)
	posDicts = addManualPosOverrides(posDicts)
	return posDicts


# ------------------------------------------------------------------
# SAVE / LOAD (ALWAYS UNDER posFolder)
# ------------------------------------------------------------------

def _ensurePosFolder() -> None:
	os.makedirs(posFolder, exist_ok=True)

def _pathInPosFolder(filename: str) -> str:
	return os.path.join(posFolder, filename)

def saveEverPosDicts(
	posDicts: Dict[str, Set[str]],
	filename: str,
	metadata: Optional[Dict[str, Any]] = None
) -> None:
	_ensurePosFolder()

	path = _pathInPosFolder(filename)

	payload = {
		"posDicts": {k: sorted(v) for k, v in posDicts.items()},
		"metadata": metadata or {}
	}

	with gzip.open(path, "wb") as f:
		pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

	metaPath = path + ".meta.json"
	with open(metaPath, "w", encoding="utf-8") as f:
		json.dump(
			{
				**payload["metadata"],
				"savedAtUnix": int(time.time()),
				"counts": {k: len(v) for k, v in posDicts.items()},
			},
			f,
			indent=2,
			sort_keys=True,
		)

def loadEverPosDicts(filename: str) -> Dict[str, Set[str]]:
	path = _pathInPosFolder(filename)
	with gzip.open(path, "rb") as f:
		payload = pickle.load(f)
	posDicts = {k: set(v) for k, v in payload["posDicts"].items()}
	posDicts = ensurePosDictsComplete(posDicts)
	return posDicts


# ------------------------------------------------------------------
# RUNTIME QUERY (FAST)
# ------------------------------------------------------------------

_POS_DICTS: Dict[str, Set[str]] = {}

def setActivePosDicts(posDicts: Dict[str, Set[str]]) -> None:
	global _POS_DICTS
	_POS_DICTS = ensurePosDictsComplete(posDicts)
	everPos.cache_clear()

@lru_cache(maxsize=300000)
def everPos(word: str, spacyPos: str) -> bool:
	result = False
	if spacyPos not in SPACY_POS:
		raise RuntimeError(f"everPos error: unsupported spacy POS type '{spacyPos}'")
	normalizedWord = normalizeWord(word)
	if spacyPos in _POS_DICTS:
		if normalizedWord in _POS_DICTS[spacyPos]:
			result = True
	if not result:
		if spacyPos == "NUM" and isNumericWord(normalizedWord):
			result = True
		elif spacyPos == "PUNCT" and isPunctWord(normalizedWord):
			result = True
		elif spacyPos == "SYM" and isSymbolWord(normalizedWord):
			result = True
	return result


# ------------------------------------------------------------------
# WRAPPER FUNCTIONS (PROJECT API)
# ------------------------------------------------------------------

def isPOSdatabaseCreated() -> bool:
	"""
	Return True if the POS database file already exists in posFolder.
	"""
	result = os.path.isfile(os.path.join(posFolder, posDictFile))
	return result


def createPOSdatabase() -> None:
	"""
	Build and save the POS database (one-time operation).
	Safe to call repeatedly; will overwrite existing files.
	"""
	posDicts = buildEverPosDicts()
	saveEverPosDicts(posDicts, posDictFile, metadata={"source": "WordNet+NLTK", "pos": list(SPACY_POS), "semantics": "ever-classifiable",},)


def loadPOSdatabase() -> None:
	"""
	Load the POS database from disk and activate it for runtime queries.
	Must be called before isWordEverInPOStypeList().
	"""
	posDicts = None
	requiresCreate = False
	if isPOSdatabaseCreated():
		posDicts = loadEverPosDicts(posDictFile)
		if posDictsMissingTypes(posDicts):
			requiresCreate = True
	else:
		requiresCreate = True
	if requiresCreate:
		createPOSdatabase()
		posDicts = loadEverPosDicts(posDictFile)
	posDicts = ensurePosDictsComplete(posDicts)
	setActivePosDicts(posDicts)

def isWordEverInPOStypeList(word: str, posTypeList) -> bool:
	"""
	Return True if `word` is EVER classifiable as ANY POS in posTypeList.

	posTypeList example: ["NOUN", "VERB"]
	"""
	result = False
	for pos in posTypeList:
		if everPos(word, pos):
			result = True
	return result

def printPOSdatabase() -> None:
	requireLoad = False
	if(_POS_DICTS is None or posDictsMissingTypes(_POS_DICTS)):
		requireLoad = True
	if(requireLoad):
		loadPOSdatabase()
	_ensurePosFolder()
	for posType in SPACY_POS:
		if(posType not in _POS_DICTS):
			raise RuntimeError(f"printPOSdatabase error: missing POS type '{posType}'")
		filePath = _pathInPosFolder("pos_" + posType + ".txt")
		with open(filePath, "w", encoding="utf-8") as f:
			for word in sorted(_POS_DICTS[posType]):
				f.write(word + "\n")


# ------------------------------------------------------------------
# MAIN (EXAMPLE)
# ------------------------------------------------------------------

if __name__ == "__main__":

	printPOSdatabase()

	# Tests
	tests = [
		("run", "VERB"),
		("run", "NOUN"),
		("blue", "ADJ"),
		("quickly", "ADV"),
		("the", "NOUN"),
	]

	for w, p in tests:
		print(w, p, everPos(w, p))
