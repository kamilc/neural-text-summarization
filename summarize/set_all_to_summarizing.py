import numpy as np

class SetAllToSummarizing(object):
    def __call__(self, sample):
        sample['mode'] = np.ones_like(sample['mode']).astype(np.float32, copy=False)

        return sample

