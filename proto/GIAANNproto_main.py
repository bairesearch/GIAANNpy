"""GIAANNproto_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:

conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
pip install networkx
pip install matplotlib
pip install yattag
pip install torch
pip install torch_geometric
pip install nltk spacy
pip install datasets
python3 -m spacy download en_core_web_sm
pip install benepar

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
import GIAANNproto_databaseNetwork
import GIAANNproto_databaseNetworkFiles
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_databaseNetworkTrain
if(useInference):
	import GIAANNproto_predictiveNetwork


# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

databaseNetworkObject = GIAANNproto_databaseNetwork.initialiseDatabaseNetwork()
databaseNetworkObject.nlp = nlp	#used by posStringToPosInt

def main():
	global sequenceCount
	GIAANNproto_databaseNetworkFiles.initialiseDatabaseFiles()
	if(useInference and inferencePredictiveNetwork and inferenceTrainPredictiveNetworkAllSequences):
		GIAANNproto_predictiveNetwork.initialisePredictiveNetwork(databaseNetworkObject)
		GIAANNproto_databaseNetwork.backupGlobalArrays(databaseNetworkObject)

	for epochIndex in range(numberEpochs):
		print("\nepochIndex = ", epochIndex)
		# Start processing the dataset
		sequenceCount = 0
		if((useInference and not inferenceTrainPredictiveNetworkAllSequences) or debugSmallDataset):
			processPrompt()
		else:
			processDataset(dataset)
		
	if(useInference and inferencePredictiveNetwork and inferenceTrainPredictiveNetworkAllSequences and inferenceSavePredictiveNetwork):
		GIAANNproto_predictiveNetwork.inferenceSavePredictiveNetwork()

def processPrompt():
	with open(inferencePromptFile, 'r', encoding='utf-8') as file:
		text = file.read()
	articleIndex = 0
	processArticle(text, articleIndex)
	
def processDataset(dataset):
	for articleIndex, article in enumerate(dataset):
		processArticle(article['text'], articleIndex)
		if sequenceCount == maxSequences:
			break

def processArticle(text, articleIndex):
	#sequences = sent_tokenize(text)
	if(ignoreNewlineCharacters):
		text = text.replace('\n', ' ')
	textParsed = nlp(text)

	if(multisentencePredictions):
		sentences = list(textParsed.sents)
		sequences = []
		for i in range(0, len(sentences), numSentencesPerSequence):
			startIndex = sentences[i].start
			endIndex = sentences[min(i + numSentencesPerSequence, len(sentences)) - 1].end
			span = textParsed[startIndex:endIndex]
			sequences.append(span)
	else:
		sequences = list(textParsed.sents)
	
	numberOfSequences = len(sequences)
	for sequenceIndex, sequence in enumerate(sequences):
		lastSequenceInPrompt = False
		if(useInference and not inferenceTrainPredictiveNetworkAllSequences):
			if(sequenceIndex == numberOfSequences-1):
				lastSequenceInPrompt = True
		if(len(sequence) <= maxSequenceLength):
			processSequence(articleIndex, sequenceIndex, sequence, lastSequenceInPrompt)
		if sequenceCount == maxSequences:
			break
			
def processSequence(articleIndex, sequenceIndex, sequence, lastSequenceInPrompt):
	global sequenceCount
	
	if(debugReloadGlobalFeatureNeuronsEverySequence):
		initialiseDatabaseNetwork()
		if(not lowMem):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_databaseNetwork.initialiseFeatureNeuronsGlobal(databaseNetworkObject.c, databaseNetworkObject.f)
	if(useInference and inferencePredictiveNetwork and inferenceTrainPredictiveNetworkAllSequences):
		if(not inferenceRetainActivationsAcrossMultipleSequences or sequenceIndex==0):	#or (articleIndex==0 and sequenceIndex==0)
			GIAANNproto_databaseNetwork.restoreGlobalArrays(databaseNetworkObject)	#restore global arrays (reset activation and time etc properties between inferencePredictiveNetworkTrainAcrossMultipleSequences:articles/sequences)
		
	print(f"Processing article: {articleIndex}, sequence: {sequenceIndex} {sequence.text}")

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

	# First pass: Extract words, lemmas, POS tags, and update concept_columns_dict and c
	conceptsFound, words, lemmas, posTags = firstPass(sequence)
	
	if(conceptsFound):
		# When usePOS is enabled, detect all possible new features in the sequence
		if not (useDedicatedFeatureLists):
			detectNewFeatures(databaseNetworkObject, words, lemmas, posTags)

		# Second pass: Create observed_columns_dict
		observedColumnsDict, observedColumnsSequenceWordIndexDict = secondPass(databaseNetworkObject, lemmas, posTags)

		# Create the sequence observed columns object
		sequenceObservedColumns = GIAANNproto_databaseNetworkTrain.SequenceObservedColumns(databaseNetworkObject, words, lemmas, observedColumnsDict, observedColumnsSequenceWordIndexDict)

		if(useInference and (inferenceTrainPredictiveNetworkAllSequences or lastSequenceInPrompt)):
			# Process each concept word in the sequence (predict)
			GIAANNproto_predictiveNetwork.processConceptWordsInference(sequenceObservedColumns, sequenceCount, sequence, sequenceSeed, sequencePredict, numSeedTokens)
		else:
			# Process each concept word in the sequence (train)
			GIAANNproto_databaseNetworkTrain.processConceptWords(sequenceObservedColumns, sequenceCount, sequence, words, lemmas, posTags)

			# Update observed columns from sequence observed columns
			sequenceObservedColumns.updateObservedColumnsWrapper()

			# Save observed columns to disk
			if(useSaveData):
				GIAANNproto_databaseNetworkFiles.saveData(databaseNetworkObject, observedColumnsDict)
				
			if(drawNetworkDuringTrain):
				# Visualize the complete graph every time a new sequence is parsed by the application.
				GIAANNproto_databaseNetworkDraw.visualizeGraph(sequenceObservedColumns, save=drawNetworkDuringTrainSave, fileName=drawNetworkDuringTrainSaveFilenamePrepend+str(sequenceIndex))

	# Break if we've reached the maximum number of sequences
	sequenceCount += 1
		
def firstPass(sequence):
	words = []
	lemmas = []
	posTags = []
	newConceptsAdded = False
	conceptsFound = False
	
	for token in sequence:
		word = token.text.lower()
		lemma = token.lemma_.lower()
		pos = token.pos_  # Part-of-speech tag
		
		if usePOS:
			if pos in nounPosTags:
				# Only assign unique concept columns for nouns
				conceptsFound, newConceptsAdded = GIAANNproto_databaseNetwork.addConceptToConceptColumnsDict(databaseNetworkObject, lemma, conceptsFound, newConceptsAdded)
		else:
			# When usePOS is disabled, assign concept columns for every new lemma encountered
			conceptsFound, newConceptsAdded = GIAANNproto_databaseNetwork.addConceptToConceptColumnsDict(databaseNetworkObject, lemma, conceptsFound, newConceptsAdded)

		words.append(word)
		lemmas.append(lemma)
		posTags.append(pos)

	# If new concept columns have been added, expand arrays as needed
	if newConceptsAdded:
		if not lowMem:
			# Expand global feature neuron arrays
			if databaseNetworkObject.globalFeatureNeurons.shape[2] < databaseNetworkObject.c:
				newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.c, databaseNetworkObject.globalFeatureNeurons.shape[3])
				if(performRedundantCoalesce):
					databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
				databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons._indices(), databaseNetworkObject.globalFeatureNeurons._values(), size=newShape, dtype=arrayType, device=deviceSparse)
				
	return conceptsFound, words, lemmas, posTags

				
def secondPass(databaseNetworkObject, lemmas, posTags):
	observedColumnsDict = {}
	observedColumnsSequenceWordIndexDict = {}
	for i, lemma in enumerate(lemmas):
		pos = posTags[i]
		if usePOS:
			if pos in nounPosTags:
				conceptIndex = databaseNetworkObject.conceptColumnsDict[lemma]
				# Load observed column from disk or create new one
				observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
				observedColumnsDict[lemma] = observedColumn
				observedColumnsSequenceWordIndexDict[i] = observedColumn
		else:
			conceptIndex = databaseNetworkObject.conceptColumnsDict[lemma]
			# Load observed column from disk or create new one
			observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			observedColumnsDict[lemma] = observedColumn
			observedColumnsSequenceWordIndexDict[i] = observedColumn
	return observedColumnsDict, observedColumnsSequenceWordIndexDict


def detectNewFeatures(databaseNetworkObject, words, lemmas, posTags):
	"""
	When usePOS mode is enabled, detect all possible new features in the sequence
	by searching for all new non-nouns in the sequence.
	"""

	numNewFeatures = 0
	for j, (wordJ, posJ) in enumerate(zip(words, posTags)):
		if(processFeatureDetection(databaseNetworkObject, j, wordJ, posTags)):
			numNewFeatures += 1

	# After processing all features, update f
	databaseNetworkObject.f += numNewFeatures

	# Now, expand arrays accordingly
	if not lowMem:
		if databaseNetworkObject.f > databaseNetworkObject.globalFeatureNeurons.shape[3]:
			extraCols = databaseNetworkObject.f - databaseNetworkObject.globalFeatureNeurons.shape[3]
			newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.globalFeatureNeurons.shape[2], databaseNetworkObject.f)
			databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
			databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons.indices(), databaseNetworkObject.globalFeatureNeurons.values(), size=newShape, dtype=arrayType, device=deviceSparse)

def processFeatureDetection(databaseNetworkObject, j, wordJ, posTags):
	"""
	Helper function to detect new features prior to processing concept words.
	"""
	
	posJ = posTags[j]
	featureWord = wordJ.lower()
	
	if usePOS:
		if posJ in nounPosTags:
			return False  # Skip nouns as features

	if featureWord not in databaseNetworkObject.conceptFeaturesDict:
		databaseNetworkObject.conceptFeaturesDict[featureWord] = len(databaseNetworkObject.conceptFeaturesDict)
		databaseNetworkObject.conceptFeaturesList.append(featureWord)
		return True
	else:
		return False
	


# Load the Wikipedia dataset using Hugging Face datasets
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)

if __name__ == "__main__":
	main()
