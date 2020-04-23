import pandas as pd
import torch
import spacy
from summarize.tensorboard_trainer import TensorboardTrainer

if 'nlp' not in vars():
    print(f"Loading Spacy model")
    nlp = spacy.load(
        "en_core_web_lg",
        disable=["tagger", "ner", "textcat"]
    )
    print(f"done")

if 'articles' not in vars():
    print(f"Loading dataset")
    articles = pd.read_parquet("data/articles-processed.parquet.gzip")
    print(f"done")

trainer = TensorboardTrainer(
    name='test-run-1',
    nlp=nlp,
    dataframe=articles,
    optimizer_class_name='Adam',
    model_args={
        'hidden_size': 128,
        'input_size': 300,
        'num_layers': 2,
    },
    optimizer_args={},
    batch_size=32,
    update_every=1,
    probability_of_mask_for_word=0.2,
    device=torch.device('cuda')
)

trainer.train()
