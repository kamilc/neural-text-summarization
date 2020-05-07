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
orig = articles
articles = orig[orig.length < 1200]#.sample(n=10)

# while len(articles[articles.set == 'test']) == 0:
#     print("resampling")
#     articles = orig[orig.length < 1500].sample(n=10)

print(f"Dataset size: {len(articles)}")

vocabulary = Vocabulary(nlp, [articles.headline], size=10000)

trainer = TensorboardTrainer(
    name='run-85',
    vocabulary=vocabulary,
    dataframe=articles,
    optimizer_class_name='Adam',
    discriminator_optimizer_class_name='Adam',
    model_args={
        'hidden_size': 512,
        'input_size': 300,
        'num_layers': 2,
        'num_heads': 10,
        'dropout_rate': 0.2,
        'dim_feedforward_transformer': 512,
        'vocabulary_size': len(vocabulary)
    },
    discriminator_args={
        'input_size': 512,
        'hidden_size': 512,
    },
    optimizer_args={
        'lr': 1e-3
    },
    discriminator_optimizer_args={
        'lr': 1e-4
    },
    batch_size=4,
    device=torch.device('cuda')
)

trainer.train(evaluate_every=1000)
