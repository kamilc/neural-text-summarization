import os
import functools
import statistics
import itertools
import random
import math
from pathlib import Path
import pdb

import pandas as pd
import swifter
import numpy as np
import hickle as hkl
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import spacy
from cached_property import cached_property

if 'nlp' not in vars():
    nlp = spacy.load(
        "en_core_web_lg",
        disable=["tagger", "ner", "textcat"]
    )

if 'articles' not in vars():
    articles = pd.read_parquet("data/articles-processed.parquet.gzip")

vocabulary = Vocabulary(nlp, [ articles["text"], articles["headline"] ])

trainer = InNotebookTrainer(
    'test-run-1',
    vocabulary,
    articles,
    optimizer_class_name='Adam',
    model_args={
        'hidden_size': 128,
        'input_size': 300,
        'num_layers': 2,
        'vocabulary_size': len(vocabulary)
    },
    optimizer_args={},
    batch_size=32,
    update_every=1,
    probability_of_mask_for_word=0.2,
    device=torch.device('cuda')
)

trainer.train()

