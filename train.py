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
    name='run-10',
    nlp=nlp,
    dataframe=articles,
    optimizer_class_name='Adam',
    model_args={
        'hidden_size': 300,
        'input_size': 300,
        'num_layers': 6,
        'num_heads': 10,
        'dropout_rate': 0.2,
        'dim_feedforward_transformer': 3072
    },
    optimizer_args={},
    batch_size=2,
    update_every=1,
    device=torch.device('cuda')
)

trainer.train()
