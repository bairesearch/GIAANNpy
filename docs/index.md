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
python -m spacy download en_core_web_trf [spacyModelName]
pip install nltk
```

## Usage
- Run the prototype from the `proto` directory.
```
source activate pytorchsenv
python GIAANNproto_main.py
```

## Development
- Review the [GIAANNproto1.nlc](https://github.com/bairesearch/GIAANNpy/blob/main/GIAANNproto1.nlc) specification for GIAANNpy requirements and design notes.
- The conceptual development of GIAANN from GIA is recorded in the [dev/](https://github.com/bairesearch/GIAANNpy/tree/main/dev) folder

## Paper
Read the current paper draft here: [GIAANN paper]({{ "/paper/" | relative_url }}) ([GIAANN-paper-WIP.pdf](https://github.com/bairesearch/GIAANNpy/releases/download/assets/GIAANN-paper-WIP.pdf)).

## Blog
ML community statements are released on the blog: [GIAANN blog]({{ "/blog/" | relative_url }}).

## Links
- [GitHub repository](https://github.com/bairesearch/GIAANNpy)
- [Issue tracker](https://github.com/bairesearch/GIAANNpy/issues)
