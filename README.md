# A Retrieval-Augmented QA System for Dungeons & Dragons

## Description

This project is a Question-Answering (QA) system that allows users to ask questions specifically tailored to the 5th edition *Dungeons & Dragons*. 

## Download

The code can be accessed and downloaded via this link: https://github.com/blsUr1/a-retrieval-augmented-qa-system-for-dungeons-and-dragons

## Usage

Examples on how to use the system can be viewed under *2.10 Usage Examples* (without filtering) and *3.3 Usage Examples* (with filtering) in `main.ipynb`.

## Requirements

This QA system was developed in *Visual Studio Code* on *Windows 11* with an *Anaconda base (Python 3.12.7)*.

It is recommended to create a new environment to avoid conflicts:

```python
conda create -n qa_env python=3.12 -y
conda activate qa_env
```

Run this code to install the required Python packages and libraries for `main.ipynb`:

```python
%pip install -U pip
%pip install datasets
%pip install huggingface_hub
%pip install fsspec==2024.6.1
%pip install s3fs==2024.6.1
%pip install evaluate
%pip install haystack-ai
%pip install google-genai-haystack
%pip install "sentence-transformers>=4.1.0"
%pip install "huggingface_hub>=0.23.0"
%pip install "transformers[torch,sentencepiece]"
%pip install "huggingface_hub[hf_xet]"
%pip install "numpy==1.26.4"
%pip install "h5py==3.10.0"
%pip install "thinc==8.2.3"

import json
import os
import re
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.joiners import DocumentJoiner
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.rankers import SentenceTransformersSimilarityRanker 
from IPython.display import clear_output
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.metrics import f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
```

Run this code to install additional Python packages and libraries for `api_to_json.ipynb`:

```python
import requests
import pprint
import json
from bs4 import BeautifulSoup
import re
import time
```

To use the LLM, a Google API Key is required. A tutorial for creating one can be found [here](https://cloud.google.com/docs/authentication/api-keys?hl=de#gcloud). Insert the key into the corresponding Python cell under subsection *2.8 LLM Initialization* in `main.ipynb`.