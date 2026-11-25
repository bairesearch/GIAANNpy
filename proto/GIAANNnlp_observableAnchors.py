"""Causal concept assignment for a fixed subword vocabulary."""

from functools import lru_cache
from types import SimpleNamespace
from GIAANNcmn_globalDefs import *
import GIAANNnlp_sequenceTokens as sequenceTokens


@lru_cache(maxsize=None)
def isIsolatedNoun(tokenId):
	result = False
	if(trainUpgradeWithObservableNounAnchors):
		import GIAANNnlp_main
		encoding = sequenceTokens.getTokeniserSubwordEncoding()
		try:
			text = encoding.decode_single_token_bytes(tokenId).decode(upgradeUTF8)
		except UnicodeDecodeError:
			# A fragment of a Unicode character cannot be an observable noun by itself.
			text = None
		if(text and encoding.encode_ordinary(text) == [tokenId]):
			tokens = sequenceTokens.getTokens(sequenceTokens.preprocessSequence(GIAANNnlp_main.nlpSequence(text)))
			result = len(tokens) == upgradeOne and tokens[upgradeZero].tokenId == tokenId and sequenceTokens.isConcept(tokens[upgradeZero])
	return result


def observableAnchor(tokens, position):
	if(trainUpgradeWithObservableNounAnchors):
		if(not tokeniserSubword or not tokeniserSubwordColumnIdentificationByFirstNounToken or not upgradeZero <= position < len(tokens)):
			raise ValueError(upgradeInvalidConfiguration)
		result = position == upgradeZero or (isIsolatedNoun(tokens[position].tokenId) and not isIsolatedNoun(tokens[position - upgradeOne].tokenId))
	return result


def buildSequence(database, tokens):
	if(inferenceUpgradePrefixOnlyEvaluation):
		if(not tokeniserSubword or not tokeniserSubwordColumnIdentificationByFirstNounToken or not tokens or numSeedTokens < upgradeOne or useTrainDuringInference or inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures or not inferenceContextLikelihood):
			raise ValueError(upgradeInvalidPrefixConfiguration)
		if(trainUpgradeWithObservableNounAnchors):
			positions = [index for index in range(len(tokens)) if observableAnchor(tokens, index)]
		else:
			positions = prefixNounPositions(tokens)
		anchors = {index: sequenceTokens.getTokeniserSubwordConceptIndex(tokens[index]) for index in positions}
		columns = []
		for position in range(len(tokens)):
			if(position in anchors):
				column = anchors[position]
			columns.append(column)
		result = SimpleNamespace(databaseNetworkObject=database, observedColumnsDict={}, columnsIndexSequenceWordIndexDict=anchors, tokenConceptColumnIndexList=columns, noDelimiterDetectedBetweenConceptTokens=False)
	return result


def prefixNounPositions(tokens):
	if(inferenceUpgradePrefixOnlyEvaluation):
		import GIAANNnlp_main
		import GIAANNnlp_sequenceConcepts
		encoding = sequenceTokens.getTokeniserSubwordEncoding()
		ids = [token.tokenId for token in tokens]
		positions = [upgradeZero]
		for doc, index in GIAANNnlp_main.nlpSequence.pipe(decodablePrefixes(encoding, ids), as_tuples=True):
			prefix = sequenceTokens.getTokens(sequenceTokens.preprocessSequence(doc))
			if([token.tokenId for token in prefix] == ids[:index + upgradeOne] and GIAANNnlp_sequenceConcepts.isTokenConceptColumnCandidate(prefix[-upgradeOne], prefix, index)):
				positions.append(index)
	return positions


def decodablePrefixes(encoding, ids):
	if(inferenceUpgradePrefixOnlyEvaluation):
		data = bytearray()
		for index, tokenId in enumerate(ids):
			data.extend(encoding.decode_single_token_bytes(tokenId))
			if(index > upgradeZero):
				try:
					text = data.decode(upgradeUTF8)
				except UnicodeDecodeError:
					text = None
				if(text is not None):
					yield text, index


def neutralTokens(database, tokens):
	if(inferenceUpgradePrefixOnlyEvaluation):
		result = []
		for token in tokens:
			name = database.conceptFeaturesList[sequenceTokens.getTokeniserSubwordFeatureIndex(token)]
			result.append(sequenceTokens.SequenceToken(name, name, upgradeNeutralPOS, upgradeNeutralPOS, token.tokenId))
	return result


def stepView(sequence, tokens, mask, position, seed):
	if(inferenceUpgradePrefixOnlyEvaluation):
		stop = position + upgradeOne if seed else position
		columns = sequence.tokenConceptColumnIndexList[:stop]
		anchors = {key: column for key, column in sequence.columnsIndexSequenceWordIndexDict.items() if key < stop}
		visible = tokens[:stop]
		known = mask[:position + upgradeOne].clone()
		if(not seed):
			# The native selector expects a slot at the forecast position. Use the last
			# observation until the selector has committed to a physical prediction.
			visible.append(visible[-upgradeOne])
			columns.append(columns[-upgradeOne])
			known[position] = False
		view = SimpleNamespace(databaseNetworkObject=sequence.databaseNetworkObject, observedColumnsDict=sequence.observedColumnsDict, columnsIndexSequenceWordIndexDict=anchors, tokenConceptColumnIndexList=columns)
		if(not seed):
			targetToken = tokens[position]
			targetColumn = sequence.tokenConceptColumnIndexList[position]
			targetAnchor = bool(mask[position])
			def revealObservation():
				if(inferenceUpgradePrefixOnlyEvaluation):
					visible[position] = targetToken
					columns[position] = targetColumn
					known[position] = targetAnchor
					if(targetAnchor):
						anchors[position] = targetColumn
				return
			view.revealObservation = revealObservation
	return view, visible, known
