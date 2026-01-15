"""GIAANNproto_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:

conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
pip install networkx
pip install matplotlib
pip install torch
pip install spacy
pip install "datasets<3" "fsspec==2024.6.1" "gcsfs==2024.6.1"
python -m spacy download spacyModelName (default:en_core_web_trf, orig: en_core_web_sm)
pip install nltk

# Usage:
source activate pytorchsenv
python GIAANNproto_main.py

# Description:
GIA ANN proto main

"""

# Import necessary libraries
import torch as pt
import spacy
from datasets import load_dataset

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_databaseNetworkExcitation
import GIAANNproto_databaseNetworkFilesExcitation
import GIAANNproto_databaseNetworkDrawExcitation
if(SANIconceptNeurons):
	import GIAANNproto_sequenceSANIconceptNeurons
import GIAANNproto_sequenceTokens
import GIAANNproto_sequenceConcepts
import GIAANNproto_sequenceObservedColumnsExcitation
import GIAANNproto_databaseNetworkTrainExcitation
if(useInference):
	import GIAANNproto_prediction

# Load the Wikipedia dataset using Hugging Face datasets
#dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True, trust_remote_code=True)

# Initialize spaCy model
nlp = spacy.load(spacyModelName)

databaseNetworkObject = GIAANNproto_databaseNetworkExcitation.initialiseDatabaseNetwork()
databaseNetworkObject.nlp = nlp	#used by posStringToPosInt

def main():
	global sequenceCount
	if(usePOS):
		GIAANNproto_sequenceTokens.loadPOSdatabase()
	GIAANNproto_databaseNetworkFilesExcitation.initialiseDatabaseFiles()
	ensurePredictiveInferenceDatabaseReady()
	if(useInference):
		if(inferenceTrainPredictiveNetworkAllSequences):
			if(inferencePredictiveNetwork):
				GIAANNproto_predictionNetwork.initialisePredictiveNetwork(databaseNetworkObject)
			GIAANNproto_databaseNetworkExcitation.backupGlobalArrays(databaseNetworkObject)
		else:
			if(not inferenceTrainFirstSequences):
				GIAANNproto_databaseNetworkExcitation.backupGlobalArrays(databaseNetworkObject)
	if(SANIconceptNeurons):
		GIAANNproto_sequenceSANIconceptNeurons.initialiseSANIconceptNeurons()
		
	for epochIndex in range(numberEpochs):
		print("\nepochIndex = ", epochIndex)
		# Start processing the dataset
		sequenceCount = 0
		if((useInference and not inferenceTrainPredictiveNetworkAllSequences) or debugSmallDataset):
			processPrompt()
		else:
			processDataset(dataset)
		
	if(useInference and inferencePredictiveNetwork and inferenceTrainPredictiveNetworkAllSequences and inferenceSavePredictiveNetwork):
		GIAANNproto_predictionNetwork.inferenceSavePredictiveNetwork()
	if(SANIconceptNeurons):
		GIAANNproto_sequenceSANIconceptNeurons.finaliseSANIconceptNeurons()

def ensurePredictiveInferenceDatabaseReady():
	if(useInference and inferenceTrainPredictiveNetworkAllSequences):
		missingFiles = []
		if not GIAANNproto_databaseNetworkFilesExcitation.pathExists(conceptColumnsDictFile):
			missingFiles.append(conceptColumnsDictFile)
		if not GIAANNproto_databaseNetworkFilesExcitation.pathExists(conceptFeaturesDictFile):
			missingFiles.append(conceptFeaturesDictFile)
		if missingFiles:
			print("error: inferenceTrainPredictiveNetworkAllSequences expects the database network to have been trained with useInference=False on all sequences.")
			print("missing database files:", ", ".join(missingFiles))
			print("precondition: expects database network to have been completely trained (with !useInference on all sequences).")
			raise SystemExit(1)

def buildSequenceWithDelimiters(sequence, tokens):
	if(not conceptColumnsDelimitByPOS):
		delimiterTypes = ["non"] * len(tokens)
	else:
		delimiterTypes = []
		for tokenIndex, token in enumerate(tokens):
			_, isDelimiterDeterministic, isDelimiterProbabilistic = GIAANNproto_sequenceConcepts.isFeaturePOSreferenceSetDelimiterType(token.word, token, tokens, tokenIndex)
			if(isDelimiterDeterministic):
				delimiterTypes.append("Dd")	#deterministic
			elif(isDelimiterProbabilistic):
				delimiterTypes.append("Di")	#indeterministic
			elif(GIAANNproto_sequenceTokens.isConcept(token)):
				delimiterTypes.append("C")	#concept
			else:
				delimiterTypes.append("")	#non
	sentenceWithDelimiters = " ".join(
		f"{token.text} ({tokenIndex}:{delimiterTypes[tokenIndex]})"
		for tokenIndex, token in enumerate(sequence)
	)
	return sentenceWithDelimiters

			
def processPrompt():
	with open(inferencePromptFile, 'r', encoding='utf-8') as file:
		text = file.read()
	articleIndex = 0
	processArticle(text, articleIndex)
	
def processDataset(dataset):
	for articleIndex, article in enumerate(dataset):
		processArticle(article['text'], articleIndex)
		if(sequenceCount == trainMaxSequences and useMaxSequences):
			break

def processArticle(text, articleIndex):
	#sequences = sent_tokenize(text)
	if(ignoreNewlineCharacters):
		text = text.replace('\n', ' ')
	textParsed = nlp(text)
	sentences = list(textParsed.sents)
	sequences = []
	sequencesRaw = []
	minSequenceLength = numSeedTokensInference + 1

	if(multisentencePredictions):
		for i in range(0, len(sentences), numSentencesPerSequence):
			startIndex = sentences[i].start
			endIndex = sentences[min(i + numSentencesPerSequence, len(sentences)) - 1].end
			span = textParsed[startIndex:endIndex]
			sequenceText = span.text
			if(not sequenceText.strip()):
				continue	#avoid whitespace-only sequences (spaCy transformer shape mismatch)
			sequenceText = sequenceText.lstrip()	#CHECKTHIS
			sequenceParsed = nlp(sequenceText)
			if(len(sequenceParsed) == 0):
				continue
			if(len(sequenceParsed) < minSequenceLength):
				continue
			sequences.append(sequenceParsed)
			sequencesRaw.append(sequenceText)
	else:
		for sentence in sentences:
			sequenceText = sentence.text
			if(not sequenceText.strip()):
				continue	#avoid whitespace-only sequences (spaCy transformer shape mismatch)
			sequenceText = sequenceText.lstrip()
			sequenceParsed = nlp(sequenceText)
			if(len(sequenceParsed) == 0):
				continue
			if(len(sequenceParsed) < minSequenceLength):
				continue
			sequences.append(sequenceParsed)
			sequencesRaw.append(sequenceText)

	numberOfSequences = len(sequences)
	for sequenceIndex, sequence in enumerate(sequences):
		sequenceRaw = sequencesRaw[sequenceIndex]
		lastSequenceInPrompt = False
		if(useInference and not inferenceTrainPredictiveNetworkAllSequences):
			if(inferenceTrainFirstSequences):
				if(sequenceIndex == numberOfSequences-1):
					print("\ninferenceTrainFirstSequences: executing inference:")
					lastSequenceInPrompt = True
				else:
					if(sequenceIndex==0):
						print("\ninferenceTrainFirstSequences: executing train:")
			else:
				if(sequenceIndex==0):
					print("\n!inferenceTrainFirstSequences: executing inference:")
				lastSequenceInPrompt = True
		if(len(sequence) <= maxSequenceLength):
			processSequence(articleIndex, sequenceIndex, sequence, sequenceRaw, lastSequenceInPrompt)
		if(sequenceCount == trainMaxSequences and useMaxSequences):
			break
			
def processSequence(articleIndex, sequenceIndex, sequence, sequenceRaw, lastSequenceInPrompt):
	global sequenceCount
	global drawRelationTypes

	sequence = GIAANNproto_sequenceTokens.preprocessSequence(sequence)
	
	if(debugReloadGlobalFeatureNeuronsEverySequence):
		initialiseDatabaseNetwork()
		if(not lowMem):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_databaseNetworkExcitation.initialiseFeatureNeuronsGlobal(databaseNetworkObject.c, databaseNetworkObject.f)
	if(useInference):
		if(inferenceTrainPredictiveNetworkAllSequences):
			if(not inferenceRetainActivationsAcrossMultipleSequences or sequenceIndex==0):	#or (articleIndex==0 and sequenceIndex==0)
				GIAANNproto_databaseNetworkExcitation.restoreGlobalArrays(databaseNetworkObject)	#restore global arrays (reset activation and time etc properties between inferencePredictiveNetworkTrainAcrossMultipleSequences:articles/sequences)
		else:
			if(not inferenceTrainFirstSequences):
				GIAANNproto_databaseNetworkExcitation.restoreGlobalArrays(databaseNetworkObject)	#reset activations so each prompt sequence is independent
	
	databaseNetworkObject.articleIndexDebug = articleIndex
	databaseNetworkObject.sequenceIndexDebug = sequenceIndex
	
	# Refresh the observed columns dictionary for each new sequence
	observedColumnsDict = {}  # key: lemma, value: ObservedColumn
	observedColumnsSequenceWordIndexDict = {}  # key: sequence word index, value: ObservedColumn
	
	if(useInference and (inferenceTrainPredictiveNetworkAllSequences or lastSequenceInPrompt)):
		if(lastSequenceInPrompt):
			if(numSeedTokens >= len(sequence)):
				return
		sequenceSeed = sequence[0:numSeedTokens]	#prompt
		sequencePredict = sequence[numSeedTokens:]

	# First pass: Extract words, lemmas, pos, tags, and update concept_columns_dict and c
	conceptsFound, conceptMask = GIAANNproto_sequenceConcepts.firstPass(databaseNetworkObject, sequence)
	
	SANIconceptNeuronSequenceList = []
	if(conceptsFound):
		if(SANIconceptNeurons):
			SANIconceptNeuronSequenceList = GIAANNproto_sequenceSANIconceptNeurons.generateSANIsequenceList(sequence, conceptMask, nlp)
		else:
			SANIconceptNeuronSequenceList.append(sequence)
		
	for sequence in SANIconceptNeuronSequenceList:
		tokens = GIAANNproto_sequenceTokens.getTokens(sequence)

		# When usePOS is enabled, detect all possible new features in the sequence
		if not (useDedicatedFeatureLists):
			GIAANNproto_sequenceConcepts.detectNewFeatures(databaseNetworkObject, tokens)

		if(debugPrintTrainSequencePOS):
			sentenceWithPOS = " ".join(f"{token.text} ({tokenIndex}:{token.pos_})" for tokenIndex, token in enumerate(sequence))
			print(f"Processing sequenceCount: {sequenceCount}, {sentenceWithPOS}")	#article: {articleIndex}, sequence: {sequenceIndex}
		if(debugPrintTrainSequenceDelimiters):
			sentenceWithDelimiters = buildSequenceWithDelimiters(sequence, tokens)
			print(f"Processing sequenceCount: {sequenceCount}, {sentenceWithDelimiters}")	#article: {articleIndex}, sequence: {sequenceIndex}
		if(debugPrintTrainSequenceRaw):
			print(sequenceRaw)
		if(debugPrintTrainSequenceDefault):
			print(f"Processing sequenceCount: {sequenceCount}, {sequence.text}")	#"{sequence.text}"	#"Processing sequenceCount: {sequenceCount}, {sequence.text}"	#article: {articleIndex}, sequence: {sequenceIndex}

		# Second pass: Create observed_columns_dict
		observedColumnsDict, observedColumnsSequenceWordIndexDict = GIAANNproto_sequenceConcepts.secondPass(databaseNetworkObject, tokens)

		# Create the sequence observed columns object
		sequenceObservedColumns = GIAANNproto_sequenceObservedColumnsExcitation.SequenceObservedColumns(databaseNetworkObject, tokens, observedColumnsDict, observedColumnsSequenceWordIndexDict)

		if(useInference and (inferenceTrainPredictiveNetworkAllSequences or lastSequenceInPrompt)):
			if(conceptColumnsDelimitByPOS and sequenceObservedColumns.noDelimiterDetectedBetweenConceptTokens):
				print("warning: inference skipped due to missing concept column delimiter detection in sequence")
			else:
				# Process each concept word in the sequence (predict)
				GIAANNproto_prediction.processConceptWordsInference(sequenceObservedColumns, sequenceCount, sequence, sequenceSeed, sequencePredict, numSeedTokens)
		else:
			# Process each concept word in the sequence (train)
			trained = GIAANNproto_databaseNetworkTrainExcitation.trainConceptWords(sequenceObservedColumns, sequenceCount, sequence, tokens)
			if(trained):
				# Update observed columns from sequence observed columns
				sequenceObservedColumns.updateObservedColumnsWrapper()

				# Save observed columns to disk
				if(useSaveData):
					GIAANNproto_databaseNetworkFilesExcitation.saveData(databaseNetworkObject, observedColumnsDict)
					
				if(drawNetworkDuringTrain):
					# Visualize the complete graph every time a new sequence is parsed by the application.
					GIAANNproto_databaseNetworkDrawExcitation.visualizeGraph(sequenceObservedColumns, False, save=drawNetworkDuringTrainSave, fileName=drawNetworkDuringTrainSaveFilenamePrepend+generateDrawSequenceIndex(sequenceIndex))

		if(debugDeleteGPUcache):
			if(pt.cuda.is_available()):
				pt.cuda.empty_cache()

	# Break if we've reached the maximum number of sequences
	sequenceCount += 1
	#note sequenceCount can be used as sequenceIndex (independent of index in sequenceList) because sequenceIndex is only used to index sequence time (same for all sequences in sequenceList)
		
if __name__ == "__main__":
	main()
