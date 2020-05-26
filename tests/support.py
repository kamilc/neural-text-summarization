import spacy
import pandas as pd
from rich import print
from cached_property import cached_property
from lib.nlp.vocabulary import Vocabulary

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class Support(object):
    @cached_property
    def nlp(self):
        print(f"[orange]- [[.]] Loading English model ...[/orange]", end="")
        model = spacy.load(
            "en_core_web_lg",
            disable=["tagger", "ner", "textcat"]
        )
        print(f"\r[green]- [[X]] Loading English model[/green]")
        return model

    @cached_property
    def articles(self):
        print(f"[orange]- [[.]] Loading dataframe ...[/orange]", end="")
        dataframe = pd.read_parquet("data/articles-processed.parquet.gzip")
        print(f"\r[green]- [[X]] Loading dataframe[/green]")
        return dataframe

    @cached_property
    def rocstories(self):
        print(f"[orange]- [[.]] Loading rocstories ...[/orange]", end="")
        dataframe = pd.read_csv("data/rocstores/dataset.csv")
        print(f"\r[green]- [[X]] Loading rocstories[/green]")
        return dataframe

    def capped_vocabulary(self, size):
        return Vocabulary(self.nlp, [self.articles.text, self.articles.headline], size=size)

    @cached_property
    def vocabulary(self):
        return Vocabulary(self.nlp, [self.articles.text, self.articles.headline])

    @cached_property
    def roc_vocabulary(self):
        return Vocabulary(self.nlp, [
            self.rocstories[f"sentence{i}"]
            for i in range(1, 6)
        ], name="roc_vocabulary")
