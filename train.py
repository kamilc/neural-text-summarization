import pandas as pd
import torch
from rich import print
import spacy
from summarize.tensorboard_trainer import TensorboardTrainer

if 'nlp' not in vars():
    print(f"[orange]- [[.]] Loading SpaCy model ...[/orange]", end="")
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
articles = articles[articles.length < 200]

trainer = TensorboardTrainer(
    name='run-17',
    nlp=nlp,
    dataframe=articles,
    optimizer_class_name='Adam',
    model_args={
        'hidden_size': 300,
        'input_size': 300,
        'num_layers': 6,
        'num_heads': 30,
        'dropout_rate': 0.2,
        'dim_feedforward_transformer': 1024
    },
    optimizer_args={},
    batch_size=2,
    update_every=1,
    device=torch.device('cuda')
)

trainer.train()
