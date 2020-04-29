import swifter

class TextToParsedDoc(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, sample):
        sample['text_doc'] = sample.swifter.progress_bar(False).apply(lambda row: self.nlp(row['text']), axis=1)
        sample['headline_doc'] = sample.swifter.progress_bar(False).apply(lambda row: self.nlp(row['headline']), axis=1)
        return sample

