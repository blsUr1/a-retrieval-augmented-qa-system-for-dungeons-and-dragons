# A Retrieval-Augmented QA System for Dungeons & Dragons

## Description

This project is a Question-Answering (QA) system that allows users to ask questions specifically tailored to *Dungeons & Dragons*.

## GitHub Repository

The code can be accessed via this link: https://github.com/blsUr1/a-retrieval-augmented-qa-system-for-dungeons-and-dragons

## Requirements

The QA system has been developed in *Visual Studio Code* on *Windows 11* with *Python 3.11* installed.

Run this code to install the required Python packages and libraries:

```python
%pip install -U datasets huggingface_hub fsspec
%pip -m spacy download en_core_web_sm
%pip install haystack-ai
%pip install google-genai-haystack
%pip install "sentence-transformers>=4.1.0"
%pip install "fsspec==2023.9.2"
%pip install "sentence-transformers>=4.1.0" "huggingface_hub>=0.23.0"
%pip install transformers[torch,sentencepiece]
%pip install huggingface_hub[hf_xet]
%pip install evaluate

import pprint
import json
import os
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.utils import Secret
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.joiners import DocumentJoiner
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.rankers import SentenceTransformersSimilarityRanker 
import evaluate
from sklearn.metrics import f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import math
import numpy as np
```





