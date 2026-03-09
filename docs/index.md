---
layout: page
title: Home
---

## Overview
GIAANNpy is a research codebase for the GIAANN project, focused on large-scale neural computation and concept-column organization.

[https://github.com/bairesearch/GIAANNpy](https://github.com/bairesearch/GIAANNpy)

## Install
- Clone the GitHub repository [GIAANNpy](https://github.com/bairesearch/GIAANNpy).
- Follow the setup steps in the repository [README](https://github.com/bairesearch/GIAANNpy/blob/main/README.md).
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
## Usage
- Run the prototype from the `proto` directory.
```
source activate pytorchsenv
python GIAANNproto_main.py
```

## Algorithm

GIA ANN is a General Intelligence Algorithm Artificial Neural Network (a neural network implementation of GIA).

GIA ANN Prototype ("proto") is a language model.

### Train

GIA ANN creates an entirely excitatory database network formed by feature neurons in columns. Upon reading a sequence during training or inference (prompt/seed) it identifies relevant concept columns, segmenting the sequence into these columns. It assigns the tokens of the trained sequence (or prompt/seed) to extant or new feature neurons in the network columns. It then forms connections between the columns.

Typically the neurons in GIA ANN are segmented into compartments (SANI: Sequentially/Segmentally Activated Neuronal Input), have multiple branches (dendrites), and are sensitive to the timing of their activated segments.

![GIAANNdemo-trainDemo-SMALL.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-trainDemo-SMALL.gif)

![GIAANNdemo-trainDemo-allColumns-SMALL.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-trainDemo-allColumns-SMALL.gif)

![understanding_the_SANI_engine_behind_GIAANN.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/understanding_the_SANI_engine_behind_GIAANN.gif)

### Inference

GIA ANN seeds the prompt provided and sequentially predicts next features in the network.

![GIAANNdemo-inferenceDemo-SMALL.gif](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANNdemo-inferenceDemo-SMALL.gif)

## Features

GIA ANN is designed to be a biologically feasible algorithm, and exhibits these properties;
* inductive bias for reasoning (generalisation from concepts).
* training speed (number of experience samples required).
* online learning (unbatched, limited precise short term memory; store in network activations themselves).
* continuous learning (dynamic update of network knowledge without compromising prior learning).
* unlimited context windows.
* biologically feasible circuitry and learning algorithm (no backpropagation).
* robustness to hallucination.

It likewise supports a number features of classical artificial neural networks such as autoregressive training/prediction and reinforcement learning.

## Configuration

All settings are located in GIAANNproto_globalDefs.py.

See the repository [README](https://github.com/bairesearch/GIAANNpy/blob/main/README.md) for a summary of the main options.

### Train/inference mode selection

#### Quick execution (demo)
For quick execution (train/inference);
* set useInference=True and inferenceTrainFirstSequences=True (and optionally drawNetworkDuringTrain=True). 

This will;
* train the database using all sequences in "database/prompt_inference.txt" except for the last (*numSentencesPerSequence) sequences, and then;
* perform inference on the last (*numSentencesPerSequence) sequences.

The prompt_inference.txt provided is taken from the first sentences from the first article of the database (Wikipedia).

#### Standard execution
* to train the network from a huggingface (current: Wikipedia) database set useInference=False.
* to perform inference on a seeded prompt (prompt_inference.txt) set useInference=True and inferenceTrainFirstSequences=False.

See the repository [README](https://github.com/bairesearch/GIAANNpy/blob/main/README.md) for more configuration details.

## Development
- Review the [GIAANNproto1.nlc](https://github.com/bairesearch/GIAANNpy/blob/main/GIAANNproto1.nlc) specification for GIAANNpy requirements and design notes.
- The conceptual development of GIAANN from GIA is recorded in the [dev/](https://github.com/bairesearch/GIAANNpy/tree/main/paper/dev) folder

## Paper
Read the current paper draft here: [GIAANN paper]({{ "/paper/" | relative_url }}) ([GIAANN-paper-WIP.pdf](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANN-paper-WIP.pdf)).

## Blog
ML community statements are released on the blog: [GIAANN blog]({{ "/blog/" | relative_url }}).

## Links
- [GitHub repository](https://github.com/bairesearch/GIAANNpy)
- [Issue tracker](https://github.com/bairesearch/GIAANNpy/issues)
