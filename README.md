# GIAANNpy

### Author

Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

### Description

General Intelligence Algorithm Artificial Neural Network (GIAANN) for Python - experimental

### License

MIT License

### Installation
```
conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
pip install networkx
pip install matplotlib
pip install torch
pip install spacy
pip install datasets
python -m spacy download spacyModelName (default:en_core_web_trf, orig: en_core_web_sm)
```

### Execution
```
source activate pytorchsenv
python GIAANNproto_main.py
```

### Overview

#### Algorithm

GIA ANN is a General Intelligence Algorithm Artificial Neural Network (a neural network implementation of GIA).

GIA ANN Prototype ("proto") is a language model.

##### Train

GIA ANN creates an entirely excitatory database network formed by feature neurons in columns. Upon reading a sequence during training or inference (prompt/seed) it identifies relevant concept columns, segmenting the sequence into these columns. It assigns the tokens of the trained sequence (or prompt/seed) to extant or new feature neurons in the network columns. It then forms connections between the columns.

Typically the neurons in GIA ANN are segmented into compartments (SANI: Sequentially/Segmentally Activated Neuronal Input), have multiple branches (dendrites), and are sensitive to the timing of their activated segments.

![GIAANNdemo-trainDemo-SMALL.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-trainDemo-SMALL.gif)

![GIAANNdemo-trainDemo-allColumns-SMALL.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-trainDemo-allColumns-SMALL.gif)

##### Inference

GIA ANN seeds the prompt provided and sequentially predicts next features in the network.

![GIAANNdemo-inferenceDemo-SMALL.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-inferenceDemo-SMALL.gif)

##### Implementation

The GIA ANN Prototype implementation is detailed in the natural language code specification (GIAANNproto*.nlc).

##### Features

GIA ANN is designed to be a biologically feasible algorithm, and exhibits these properties;
* inductive bias for reasoning (generalisation from concepts).
* training speed (number of experience samples required).
* online learning (unbatched, limited precise short term memory; store in network activations themselves).
* continuous learning (dynamic update of network knowledge without compromising prior learning).
* unlimited context windows.
* biologically feasible circuitry and learning algorithm (no backpropagation).

It likewise supports a number features of classical artificial neural networks such as autoregressive training/prediction and reinforcement learning.

### Configuration

All settings are located in GIAANNproto_globalDefs.py.

#### Train/inference mode selection

##### Quick execution (demo)
For quick execution (train/inference);
* set useInference=True and inferenceTrainFirstSequences=True (and optionally drawNetworkDuringTrain=True). 

This will;
* train the database using all sequences in "database/prompt_inference.txt" except for the last (*numSentencesPerSequence) sequences, and then;
* perform inference on the last (*numSentencesPerSequence) sequences.

The prompt_inference.txt provided is taken from the first sentences from the first article of the database (Wikipedia).

##### Standard execution
* to train the network from a huggingface (current: Wikipedia) database set useInference=False.
* to perform inference on a seeded prompt (prompt_inference.txt) set useInference=True and inferenceTrainFirstSequences=False.

#### Dataset

* maxSequenceLength = 100 - depends on CPU/GPU RAM availability during train 
* set trainMaxSequences = 10000 - max sequences for train or inference
* databaseFolder = "../database/" - set to local SSD for fast i/o

#### Multisentence predictions

* set multisentencePredictions - each sequence comprises multiple sentences

#### RAM

* set useGPUdense=True (and useGPUsparse=True) during train
* set useGPUsparse=False during inference if CPU has more RAM

#### Segment activation time

* set inferenceUseNeuronFeaturePropertiesTime=True (and inferenceUseNeuronFeaturePropertiesTimeExact=True for most strict selection) - record segment activation times and use them to bias feature selection during inference based on their proximity to their ideal (i.e. trained) timings.

#### Dendritic branches

* set multipleDendriticBranches=True to support cases where a trained sequence has repeated references to a column feature.

#### Inhibitory neurons

* Deprecated - inhibition is already simulated during topk selection during inference

#### Array properties

* set arrayIndexPropertiesEfficient=True to reduce train time/RAM (not compatible with drawRelationTypes)

#### SANI

* set useSANI=True - enables sequentially activated neuronal input

#### Immediate (direct) connections

* set enforceDirectConnections=True (by default it uses useSANI to enforce direct connections between predicted features).

#### Concept column delimiters

* POS types used to assign tokens to concept columns

#### Connection strength modifiers

* Experimental

#### Beam search

* set inferenceBeamSearch=True - executes inference by identifying the best beam path

#### SANI concept neuron

* Deprecated - execute preprocessor to allocate neurons to non-noun tuples for each concept

#### Inference

Settings for inference. See:
* inferenceConnectionsStrengthBoolean, inferenceSegmentActivationsBoolean, inferenceSourceActivationsBoolean

#### Predictive network

* Deprecated - maintain separate neural network for prediction phase

#### Train optimisations

* trainSequenceObservedColumnsUseSequenceFeaturesOnly=True - only loads sequence features into dense tensors for train. Currently uses dense tensors (typically in GPU RAM) however to merge these with database network. Can be upgraded so only a limited amount of data is ever loaded to GPU during train (it currently temporarily masks entire feature arrays in GPU during transfer phase).
* trainSequenceObservedColumnsMatchSequenceWords=True is now mandatory (originally GIAANN proto was not guaranteed to independently train a feature for every token instance in the sequence).

#### Draw

* drawSegments - draw independent segments (connections) in different colours.
* drawBranches - draw independent branches (connections and features) in different colours.
* drawRelationTypes - draw feature POS types (and their connections) in different colours.
* drawDelimiters - draws feature neuron column delimiters (and their external connections) in different colours.
* drawDefault - draws concept/feature node types (and their internal/external connections) in different colours. If useInference=True and in inferenceMode then will draw activation status of network.

##### drawSegments
![GIAANNdemo-drawSegments-SMALL.png](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-drawSegments-SMALL.png)

##### drawBranches
![GIAANNdemo-drawBranches-SMALL.png](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-drawBranches-SMALL.png)

##### drawRelationTypes
![GIAANNdemo-drawRelationTypes-SMALL.png](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-drawRelationTypes-SMALL.png)

##### drawDelimiterTypes
![GIAANNdemo-drawDelimiters-SMALL.png](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-drawDelimiters-SMALL.png)

##### drawDefault
![GIAANNdemo-drawDefault-SMALL.png](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-drawDefault-SMALL.png)


#### Algorithm preferences (normalisation, permanence etc)

* Experimental 

#### Database save paths

* Location of database files

#### SANI settings

* useSANIcolumns - assign segments by concept column proximity to connection target during train.
* useSANIfeatures - assign segments by feature proximity to connection target during train.
* useSANIfeaturesAndColumns - assign segments by column proximity first then feature proximity.
* algorithmMatrixSANImethod="enforceActivationAcrossSegments" - only activate a segment under conditions.
* algorithmMatrixSANIenforceRequirement="enforceLastSegmentMustBeActive" - only activate neuron if last segment active.
* enforceSequentialActivation - only activate a segment if previous segment was active.

### Limitations

GIAANN proto currently uses a POS tagger for 'reference set delimiter' identification (token column assignment), which itself may rely on machine learning technology, although this can be replaced with a more rudimentary algorithm.

Current implementation experiences significant slow-down and RAM usage during inference as the size of the trained database network increases (due to SSD data i/o and sparse tensor hardware acceleration limitations).


### References

* General Artificial Intelligence Method and Computer System (AU provisional) - 28 Mar 2012 (App No. 2012901230) - [Text](https://sourceforge.net/projects/opengia/files/algorithm/BAI_GeneralAIPatentAUProv1b_TextOnly_9Mar2012.pdf)/[Figures](https://sourceforge.net/projects/opengia/files/algorithm/BAI_GeneralAIPatentAUProv1b_FiguresOnly_9Mar2012.pdf)
* General Intelligence Algorithm (https://github.com/bairesearch/GIA) - [Wiki](https://github.com/bairesearch/GIA/wiki)
* Hopfield Natural Language Processing (https://github.com/bairesearch/HFNLPpy)
* Sequentially Activated Neuronal Input neural network (https://github.com/bairesearch/SANI)
* Excitatory Inhibitory Sequentially/Segmentally Activated Neuronal Input network (https://github.com/bairesearch/EISANIpt)
* Sequentially Activated Neuronal Input natural language processing (https://github.com/bairesearch/SANINLPtf)
* Local Connectome (https://github.com/bairesearch/LocalConnectome)
* Simulated Dendritic Branch artificial neural network (https://github.com/bairesearch/SDBANNtf)
* Simulated Dendritic Branch natural language processing (https://github.com/bairesearch/SDBNLPpt)
* Jamali, M., Grannan, B., Cai, J., Khanna, A.R., Munoz, W., Caprara, I., Paulk, A.C., Cash, S.S., Fedorenko, E. and Williams, Z.M. (2024). Semantic encoding during language comprehension at single-cell resolution. Nature, 1-7.
* Hawkins, J. et al. (2011). Hierarchical Temporal Memory (HTM) Whitepaper (Version 0.2.1). Numenta.
* Holtmaat, A., & Caroni, P. (2016). Functional and structural underpinnings of neuronal assembly formation in learning. Nature neuroscience, 19(12), 1553-1562.
* Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8), 2554-2558.
* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
