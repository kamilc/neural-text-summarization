import numpy as np

class AddNoiseToEmbeddings(object):
    def __init__(self, probability_of_mask_for_word):
        self.probability_of_mask_for_word = probability_of_mask_for_word
        self.rng = np.random.default_rng()

    def mask_vector(self, vector):
        """
        Masks words with zeros randomly
        """
        seq_len = vector.shape[0]
        vector_len = vector.shape[1]

        mask = np.repeat(
            self.rng.choice(
                [0, 1],
                seq_len,
                p=[
                    self.probability_of_mask_for_word,
                    (1 - self.probability_of_mask_for_word)
                ]
            ).reshape((seq_len, 1)),
            vector_len,
            axis=1
        )

        return vector * mask

    def __call__(self, sample):
        sample['noisy_word_embeddings'] = sample['word_embeddings'].apply(self.mask_vector)

        return sample
