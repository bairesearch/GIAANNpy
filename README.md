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
python -m pip install --upgrade pip
pip install networkx
pip install matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install spacy
datasetsLibrary4plus=False: pip install "datasets<4" "fsspec==2024.6.1" "gcsfs==2024.6.1"
python -m spacy download en_core_web_sm [spacyModelName]
pip install nltk
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

![understanding_the_SANI_engine_behind_GIAANN.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/understanding_the_SANI_engine_behind_GIAANN.gif)

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
* robustness to hallucination.

It likewise supports a number features of classical artificial neural networks such as autoregressive training/prediction and reinforcement learning.

### Configuration

All settings are located in `proto/GIAANNproto_globalDefs.py`.

#### Train/inference mode selection

##### Quick execution (demo)

For quick execution (train and inference);
* set `useQuickExecution = True`

This will;
* automatically set `executionMode="inference"` and `inferenceTrainFirstSequences=True` 
* train the database using all sequences in `"database/inference_prompt.txt.trainAndInference"` except for the last (`*numSentencesPerSequence`) sequences, and then;
* perform inference on the last (`*numSentencesPerSequence`) sequences.

The `database/inference_prompt.txt.trainAndInference` provided is taken from the first sentences from the first article of the dataset (Wikipedia).

##### Standard execution

For standard execution (train or inference);
* set `useQuickExecution = False`
* set `executionMode="train"` to train the network from a huggingface dataset (e.g. `Wikipedia/OSCAR-2201`), or;
* set `executionMode="inference"` to perform inference on a seeded prompt (`prompt_inference.txt.*`)

#### Primary Draw settings

* `drawNetworkDuringTrain` - draw network during train
* `drawNetworkDuringInference` - draw network during inference

#### Inference settings

* `numSeedTokensInference` - the number of tokens used for the seed (vs prediction) phase of inference. Note arrayNumberOfSegments is derived from this parameter.
* `inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False` for benchmarking top-1 accuracy (fires targets rather than predictions).
* `useBenchmarkDefaultsEvalTestSet` - select default inference settings for eval using training-set or test-set data
* `inferenceEvaluateTestSet` -  eval using training-set or test-set data
* `inferenceSegmentTiming` - "none"/"biased"/"seq"/"exact" (from least restrictive to most restrictive):
  * `"none"`: no segment timing checks.
  * `"biased`": no sequentiality enforcement but timing bias (wrt expected/train timings). 
  * `"seq"`: sequentiality enforcement only (no other timing checks). 
  * `"exact"`: sequentiality and timing enforcement (wrt expected/train timings)
* `inferenceActivationsType`: 
  * `"boolf"`: inference segment activations boolean for feature segments only
  * `"boolf+c"`: inference segment activations boolean for feature and column segments
  * `"intf+c"`: inference segment activations integer for feature and column segments

#### Dataset Type

* `datasetType` - set `"oscar"` / `"wikipedia"` / `"textfile"` [experimental]

#### Database

* `databaseFolderBase` - select local SSD for fast i/o
* `trainMaxSequences` - max sequences for train
* `maxSequenceLength` - depends on CPU/GPU RAM availability during train 

#### Multisentence predictions

* `multisentencePredictions` - each sequence comprises multiple sentences
* `numSentencesPerSequence` - the number of sentences per sequence

#### Dendritic branches

* `multipleDendriticBranches` - support cases where a trained sequence has repeated references to a column feature 
* `numberOfDendriticBranches` - number of dendritic branches
* `randomlyAssignBranches` to support increasingly conflicting reuse of phrases throughout dataset

#### Dataset

* `datasetsLibrary4plus` - selects compatible dataset for datasets library
* `datasetName` - wikipedia dataset name
* `datasetCfg` - wikipedia dataset cfg
* `useLocalDataset` - use local dataset (else stream)
* `datasetFolder` - folder to store dataset

#### RAM

* `useGPUdense=True` (and useGPUsparse=True) during train (sequence size dependent)
* `useGPUsparse=False` during inference (network size dependent)
* `storeDatabaseInRam`
* `useGPUdatabase=False` (assume CPU has more RAM)
		
#### Segment activation time

* `inferenceUseNeuronFeaturePropertiesTime` - record segment activation times and use them to bias feature selection during inference based on their proximity to their ideal (i.e. trained) timings
* `inferenceUseNeuronFeaturePropertiesTimeExact` - most strict time selection

#### Array properties

* `arrayIndexPropertiesEfficient=True` to reduce train time/RAM (not compatible with `drawRelationTypes`)

#### SANI

* `useSANI=True` - enables sequentially activated neuronal input

#### Immediate (direct) connections

* `enforceDirectConnections=True`(by default it uses useSANI to enforce direct connections between predicted features).

#### Concept column delimiters

* POS types used to assign tokens to concept columns

#### Connection strength modifiers

* Experimental

#### Beam search

* `inferenceBeamSearch` - executes inference by identifying the best beam path
* `inferenceBeamWidth` - width of beam search
* `inferenceBeamDepth` - depth of beam search

#### Inference activations

* `inferenceConnectionsStrengthBoolean`
* `inferenceSegmentActivationsBoolean`
* `inferenceSegmentActivationsBooleanFeatureSegmentsOnly`
* `inferenceSourceActivationsBoolean`

#### Train optimisations

* `trainSequenceObservedColumnsUseSequenceFeaturesOnly=True` - only loads sequence features into dense tensors for train.
* `trainSequenceObservedColumnsMatchSequenceWords=True` is now mandatory (originally GIAANN proto was not guaranteed to independently train a feature for every token instance in the sequence).

#### Draw

* `drawSegments` - draw independent segments (connections) in different colours.
* `drawBranches` - draw independent branches (connections and features) in different colours.
* `drawRelationTypes` - draw feature POS types (and their connections) in different colours.
* `drawDelimiters` - draws feature neuron column delimiters (and their external connections) in different colours.
* `drawDefault` - draws prime concept feature/instance feature node types (and their internal/external connections) in different colours. If `executionMode="inference"` or `executionMode="trainAndInference"` and in inference phase then will draw activation status of network.

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

* `useSANIcolumns` - assign segments by concept column proximity to connection target during train.
* `useSANIfeatures` - assign segments by feature proximity to connection target during train.
* `useSANIfeaturesAndColumns` - assign segments by column proximity first then feature proximity.
* `algorithmMatrixSANImethod="enforceActivationAcrossSegments"` - only activate a segment under conditions.
* `algorithmMatrixSANIenforceRequirement="enforceLastSegmentMustBeActive"` - only activate neuron if last segment active.
* `enforceSequentialActivation` - only activate a segment if previous segment was active.

#### POS

* `useSpacyForConceptNounPOSdetection` - use spacy for prime concept feature identification (dynamic context dependent pos detection), else use GIAANNproto_sequencePOS.
* 'reference set delimiter' identification (token column assignment) uses predetermined word-POS dictionary (GIAANNproto_sequencePOS).

### Paper

[GIAANN paper (PDF) - WIP](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANN-paper-WIP.pdf)

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

### Additional Execution Modes

#### Independent Database Draw execution

##### Generate the database
edit GIAANNproto_globalDefs.py;
* set `useDrawNetworkIndependently=False`
* set `executionMode="train"`
* select the database folder, e.g. `databaseFolderBase = "../database"`
* `python GIAANNproto_main.py`

##### Generate the database svg/ldr files
* set `useDrawNetworkIndependently=True`
* select the database folder, e.g. `databaseFolderBase = "../database"`
* select the `drawEfficient` settings:
  * `drawEfficientFormat3D` - save standalone drawEfficient large-network output in LDraw .ldr format instead of 2D matplotlib .svg output
    * `drawEfficientFormat3Dprism` - position standalone drawEfficient 3D columns on a square 2D grid and draw each column as a rectangular prism
  *	`drawEfficientIntracolumnHorizontalOffset` - feature neurons within columns have a horizontal x (or xy) offset applied
  * `drawEfficientDrawDeadNeurons` - draw empty columns with no connected neurons
  * `drawEfficientGrid` - draws column feature neuron y positions at their real featureIndex
  ` drawEfficientCompact`	- emulates the original draw visualisation of drawEfficient=False (but still not the same as no randomised horizontal position of nodes within columns)
* `python GIAANNproto_main.py`

##### 2D visualisation (drawEfficientFormat3D=False)

* open `database/GIAANNproto1xAllColumnsDraw.svg` using any svg viewer, e.g. Firefox.

##### 3D visualisation (drawEfficientFormat3D=True)

###### Install ldr_wgpu (GPU accelerated LDR file viewer for large files)
```
sudo snap install --classic rustup
rustup install stable
rustup default stable
sudo apt install git build-essential pkg-config
git clone -b branch-lineRendering --single-branch https://github.com/bairesearch/ldr_wgpu.git
```

###### Install LDRAW parts library
```
wget https://library.ldraw.org/library/updates/complete.zip
unzip complete.zip
mv GIAANNpy/database/4-4CUBE.DAT ldraw/parts/4-4cube.dat
```

###### Run ldr_wgpu on an LDR file
```
cd ldr_wgpu
cargo run --release -p ldr_viewer "../ldraw" "../GIAANNpy/database/GIAANNproto1xAllColumnsDraw.ldr"
```

