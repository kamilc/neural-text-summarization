"""Usage: train.py <experiment_toml_file>

Arguments:
  experiment_toml_file        path to the toml file describing the train ing experiment to run
"""

from docopt import docopt
import pandas as pd
import torch
import toml
from rich import print
import spacy
from summarize.tensorboard_trainer import TensorboardTrainer
from lib.nlp.vocabulary import Vocabulary

arguments = docopt(__doc__)

# TODO: refactor this to be under summarize
from tests.support import Support

support = Support()

# TODO: add command args parsing and taking the path here as an argument
config = toml.load(arguments['<experiment_toml_file>'])

print(f"{config['experiment']}\n\n")

config['experiment']['vocabulary'] = getattr(support, config['experiment']['vocabulary'])
config['experiment']['dataframe'] = getattr(support, config['experiment']['dataframe'])
config['experiment']['device'] = torch.device(config['experiment']['device'])

print(f"Dataset size: {len(config['experiment']['dataframe'])}")

trainer = TensorboardTrainer(**config['experiment'])
trainer.train(evaluate_every=400)
