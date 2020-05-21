import pandas as pd
import torch
import toml
from rich import print
import spacy
from summarize.tensorboard_trainer import TensorboardTrainer
from lib.nlp.vocabulary import Vocabulary

# TODO: refactor this to be under summarize
from tests.support import Support

support = Support()

# TODO: add command args parsing and taking the path here as an argument
config = toml.load("experiments/rocstories-1.toml")

print(f"{config['experiment']}\n\n")

config['experiment']['vocabulary'] = getattr(support, config['experiment']['vocabulary'])
config['experiment']['dataframe'] = getattr(support, config['experiment']['dataframe'])
config['experiment']['device'] = torch.device(config['experiment']['device'])

print(f"Dataset size: {len(config['experiment']['dataframe'])}")

trainer = TensorboardTrainer(**config['experiment'])
trainer.train(evaluate_every=400)
