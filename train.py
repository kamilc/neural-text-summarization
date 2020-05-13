import pandas as pd
import torch
import toml
from rich import print
import spacy
from summarize.tensorboard_trainer import TensorboardTrainer
from lib.nlp.vocabulary import Vocabulary

# TODO: refactor this to be under summarize
from tests.support import Support

# if 'nlp' not in vars():
#     print(f"[orange]- [[.]] Loading language ...[/orange]", end="")
#     nlp = spacy.load(
#         "en_core_web_lg",
#         disable=["tagger", "ner", "textcat"]
#     )
#     print(f"\r[green]- [[X]] Loading SpaCy model[/green]")

# if 'articles' not in vars():
#     print(f"[orange]- [[.]] Loading dataset ...[/orange]", end="")
#     articles = pd.read_parquet("data/articles-processed.parquet.gzip")
#     print(f"\r[green]- [[X]] Loading dataset[/green]")

# articles['length'] = articles.apply(lambda row: len(row['text']), axis=1)
# orig = articles
# articles = orig[orig.length < 1200]#.sample(n=100)

# while len(articles[articles.set == 'test']) == 0:
#     print("resampling")
#     articles = orig[orig.length < 1500].sample(n=100)

# print(f"Dataset size: {len(articles)}")

# vocabulary = Vocabulary(nlp, [articles.headline], size=10000)

support = Support()

config = toml.load("experiments/rocstories-1.toml")

print(f"{config['experiment']}\n\n")

config['experiment']['vocabulary'] = getattr(support, config['experiment']['vocabulary'])
config['experiment']['dataframe'] = getattr(support, config['experiment']['dataframe'])
config['experiment']['device'] = torch.device(config['experiment']['device'])

print(f"Dataset size: {len(config['experiment']['dataframe'])}")

trainer = TensorboardTrainer(**config['experiment'])
trainer.train(evaluate_every=400)

# trainer = TensorboardTrainer(
#     name='run-98',
#     vocabulary=support.roc_vocabulary,
#     dataframe=support.rocstories,
#     optimizer_class_name='Adam',
#     discriminator_optimizer_class_name='Adam',
#     model_args={
#         'hidden_size': 1024,
#         'input_size': 300,
#         'num_layers': 6,
#         'num_heads': 10,
#         'dropout_rate': 0.1,
#         'dim_feedforward_transformer': 2024,
#         'vocabulary_size': len(vocabulary)
#     },
#     discriminator_args={
#         'input_size': 1024,
#         'hidden_size': 1024,
#     },
#     optimizer_args={
#         'lr': 1e-5
#     },
#     discriminator_optimizer_args={
#         'lr': 1e-4
#     },
#     batch_size=8,
#     device=torch.device('cuda')
# )

# trainer.train(evaluate_every=1000)
