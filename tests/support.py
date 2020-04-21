import spacy
import pandas as pd
from cached_property import cached_property

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
        print(f"Loading Spacy model")
        return spacy.load(
            "en_core_web_lg",
            disable=["tagger", "ner", "textcat"]
        )

    @cached_property
    def articles(self):
        return pd.read_parquet("data/articles-processed.parquet.gzip")
