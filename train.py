import pandas as pd
import torch
from rich import print
import spacy
from summarize.tensorboard_trainer import TensorboardTrainer
from lib.nlp.vocabulary import Vocabulary

if 'nlp' not in vars():
    print(f"[orange]- [[.]] Loading language ...[/orange]", end="")
    nlp = spacy.load(
        "en_core_web_lg",
        disable=["tagger", "ner", "textcat"]
    )
    print(f"\r[green]- [[X]] Loading SpaCy model[/green]")

if 'articles' not in vars():
    print(f"[orange]- [[.]] Loading dataset ...[/orange]", end="")
    articles = pd.read_parquet("data/articles-processed.parquet.gzip")
    print(f"\r[green]- [[X]] Loading dataset[/green]")

articles['length'] = articles.apply(lambda row: len(row['text']), axis=1)
articles = articles[articles.length < 1500] #.sample(n=100)

vocabulary = Vocabulary(nlp, [articles.headline], size=10000)

trainer = TensorboardTrainer(
    name='run-34',
    vocabulary=vocabulary,
    dataframe=articles,
    optimizer_class_name='Adam',
    model_args={
        'hidden_size': 128,
        'input_size': 300,
        'num_layers': 4,
        'num_heads': 10,
        'dropout_rate': 0.2,
        'dim_feedforward_transformer': 1024,
        'vocabulary_size': len(vocabulary)
    },
    optimizer_args={},
    batch_size=4,
    update_every=1,
    device=torch.device('cuda')
)

trainer.train()
