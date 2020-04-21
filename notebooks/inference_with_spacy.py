import torch
indices = torch.randint(0,5, size=(4))
indices = torch.randint(0,5, size=(4, 7))
torch.randint(0,5, size=(4, 7))
torch.randint(0,5, size=(4,))
indices = torch.randint(0,5, size=(4,))
torch.nn.functional.one_hot(indices, 4)
indices = torch.randint(0,5000, size=(4,))
indices
torch.nn.functional.one_hot(indices, 5000)
torch.zeros((100, 10))
data = torch.zeros((100, 10))
data
data.to_sparse()
sparse = data.to_sparse()
data.shape
sparse.reshape((200, 10))
import spacy
nlp = spacy.load(
    "en_core_web_lg",
    disable=["tagger", "ner", "textcat"]
)
nlp.vocab
nlp.vocab.words
nlp.vocab.vectors
nlp.vocab.vectors[0]
nlp.vocab.vectors['yyy']
nlp.vocab.vectors['word']
nlp.vocab['word']
nlp.vocab['word'].vector
v1 = nlp.vocab['word'].vector
v2 = nlp.vocab['bird'].vector
v1 - v2
(v1 - v2[0])
(v1 - v2)[0]
v1 = nlp.vocab['bird'].vector
v2 = nlp.vocab['birdie'].vector
(v1 - v2)[0]
v1
v2
from summarize.summarize_net import SummarizeNet
var1 = SummarizeNet
var1
var1()
var1(1,2,3)
SummarizeNet.load
var1.load
word_embeddings = np.random((32, 100, 300))
import numpy as np
word_embeddings = np.random((32, 100, 300))
word_embeddings = np.random.rand((32, 100, 300))
word_embeddings = np.random.rand(32, 100, 300)
word_embeddings
nlp
nlp.vocab.vectors.most_similar(word_embeddings)
%time
%time 2+2
nlp.vectors
nlp.vocab.vectors
nlp.vocab.vectors.data
nlp.vocab.vectors.data = torch.Tensor(nlp.vocab.vectors.data).cuda()
%time nlp.vocab.vectors.most_similar(word_embeddings)
%time nlp.vocab.vectors.most_similar(torch.Tensor(word_embeddings).cuda())
torch.Tensor(word_embeddings).cuda()
%time nlp.vocab.vectors.most_similar(torch.Tensor(word_embeddings).cuda())
%time nlp.vocab.vectors.most_similar(torch.Tensor(word_embeddings, dtype=torch.float32).cuda())
%time nlp.vocab.vectors.most_similar(torch.from_numpy(word_embeddings, dtype=torch.float32).cuda())
%time nlp.vocab.vectors.most_similar(torch.from_numpy(word_embeddings).cuda())
%time nlp.vocab.vectors.most_similar(torch.from_numpy(word_embeddings).cuda())
torch.from_numpy(word_embeddings)
torch.from_numpy(word_embeddings).cuda()
nlp.vocab.vectors.data
nlp.vocab.vectors.data.as(torch.float32)
nlp.vocab.vectors.data.float()
 nlp.vocab.vectors.data.float().dtype
 nlp.vocab.vectors.data.dtype
%time nlp.vocab.vectors.most_similar(torch.from_numpy(word_embeddings).cuda())
torch.from_numpy(word_embeddings).cuda().dtype
torch.from_numpy(word_embeddings[0, :, :]).cuda().dtype
%time nlp.vocab.vectors.most_similar(torch.from_numpy(word_embeddings[0, :, :]).cuda())
%pdb
%time nlp.vocab.vectors.most_similar(torch.from_numpy(word_embeddings[0, :, :]).cuda())
%time nlp.vocab.vectors
%time nlp.vocab.vectors.data
F
%time F.cosine_similarity(nlp.vocab.vectors.data, torch.zeros_like(nlp.vocab.vectors.data))
F.cosine_similarity(nlp.vocab.vectors.data, torch.zeros_like(nlp.vocab.vectors.data))
F.cosine_similarity(nlp.vocab.vectors.data, torch.zeros_like(nlp.vocab.vectors.data)).shape
nlp.vocab.vectors.data.shape
torch.eye
torch.eye(300)
torch.eye(300)[0, :]
torch.eye(300)[1, :]
torch.eye(300)[2, :]
_eye = torch.eye(300)
_eye[0, :]
_eye[3, :]
_eye[3, :].repeat(100)
_eye[3, :].repeat(100, 300)
_eye[1, :].repeat(100, 300)
len(nlp.vocab.vectors)
len(nlp.vocab.vectors)
torch.Tensor([ ix for ix in range(0, len(nlp.vocab.vectors)) ])
torch.Tensor([ ix for ix in range(0, len(nlp.vocab.vectors)) ])
torch.Tensor([ ix for ix in range(0, len(nlp.vocab.vectors)) ])
F.cosine_similarity(nlp.vocab.vectors.data, torch.zeros_like(nlp.vocab.vectors.data))
torch.Tensor([ _eye[ix, :].repeat(100, 300) for ix in range(0, len(nlp.vocab.vectors)) ])
torch.Tensor([ F.cosine_similarity(nlp.vocab.vectors.data, _eye[ix, :].repeat(len(nlp.vocab.vectors), 300)) for ix in range(0, len(nlp.vocab.vectors)) ])
F.cosine_similarity(nlp.vocab.vectors.data, _eye[1, :].repeat(len(nlp.vocab.vectors), 300))
_eye[1, :].repeat(len(nlp.vocab.vectors), 300)
len(nlp.vocab.vectors)
_eye[1, :].repeat(len(nlp.vocab.vectors), 300)
selector = torch.zeros_like(nlp.vocab.vectors.data)
selector
selector[:,1] = 1
selector
F.cosine_similarity(nlp.vocab.vectors.data, selector)
selector[:,:] = 0
selector
selector[:,2] = 1
F.cosine_similarity(nlp.vocab.vectors.data, selector)
for ix in range(0, 300):
    selector[:,:] = 0
    selector[:,ix] = 1
    distances.append(F.cosine_similarity(nlp.vocab.vectors.data, selector))
distances = []
for ix in range(0, 300):
    selector[:,:] = 0
    selector[:,ix] = 1
    distances.append(F.cosine_similarity(nlp.vocab.vectors.data, selector))
len(distances)
distances_tensor = torch.stack(distances)
distances_tensor
distances_tensor.shape
distances_tensor[0, 0]
distances_tensor[0, 2]
distances_tensor[0, :]
distances_tensor[0, :].argmin()
distances_tensor[0, 69957]
(distances_tensor**2)[0, 69957]
(distances_tensor[0, :]**2).argmin()
word_embedding = nlp.vocab['house'].vector
word_embedding
selector.shape
word_embedding.shape
torch.stack([ l.vector for l in nlp("this is a sentence") ])
word_embeddings = torch.stack([ l.vector for l in nlp("this is a sentence") ])
word_embeddings.shape
selector_words = torch.zeros_like(word_embeddings)
distances_words = []
for ix in range(0, 300):
    selector_words[:,:] = 0
    selector_words[:,ix] = 1
    distances_words.append(F.cosine_similarity(word_embeddings, selector_words))
distances_words
distances_words.shape
distances_words_tensor = torch.stack(distances_words)
distances_words_tensor
distances_words_tensor.shape
distances_tensor
distances_tensor.shape
distances_tensor - distances_words_tensor
distances_words_tensor[0]
distances_words_tensor[0, :]
distances_tensor - distances_words_tensor[0,:]
distances_tensor.shape
distances_tensor.transpose(1, 0).shape
distances_tensor.transpose(1, 0)
distances_words_tensor[0,:]
distances_words_tensor[0,:].shape
distances_words_tensor[:, 0]
distances_tensor.shape
distances_words_tensor[0, :]
distances_tensor - distances_words_tensor[:,0]
distances_tensor.shape
distances_words_tensor[:,0].shape
distances_words_tensor.shape
distances_tensor.shape
distances_tensor.transpose(1, 0) - distances_words_tensor[:,0]
diffs = (distances_tensor.transpose(1, 0) - distances_words_tensor[:,0])**2
diffs.shape
diffs
diffs.sum(dim=1)
diffs.sum(dim=1).shape
diffs.sum(dim=1).argmin()
nlp.vocab[403305]
nlp.vocab.vectors
nlp.vocab.vectors.keys[403305]
nlp.vocab.vectors.keys
nlp.vocab.vectors.keys()
vector_keys = nlp.vocab.vectors.keys()
vector_keys[403305]
vector_keys = list(nlp.vocab.vectors.keys())
vector_keys[403305]
nlp.vocab[7618417936409410543]
nlp.vocab[7618417936409410543].text
len(vector_keys)
nlp.vocab.vectors.data[0]
vector_keys[0]
nlp.vocab[3424551750583975941]
nlp.vocab[3424551750583975941].text
nlp.vocab.vectors.keys()
nlp.vocab.vectors.keys()[0]
nlp.vocab.vectors.keys
i = 0
for key in nlp.vocab.vectors.keys():
    if i < 10:
        print(key)
        i += 1
nlp.vocab.lookups
nlp.vocab.lookups.tables
nlp.vocab.lookups.tables['lemma_index']
nlp.vocab.lookups.tables[2]
nlp.vocab.lookups['lemma_index']
i = 0
for key, value in nlp.vocab.vectors.items():
    print(key)
    i += 10
    if i > 10:
        break
nlp.vocab.strings
nlp.vocab.strings[3424551750583975941]
nlp.vocab.strings[4645222075005403145]
nlp.vocab.vectors
nlp.vocab.vectors.data[0, :]
nlp.vocab['croup']
nlp.vocab['croup'].vector
nlp.vocab.vectors
nlp.vocab.vectors.key2row
nlp.vocab['hello']
nlp.vocab['hello'].orth
nlp.vocab[5983625672228268878]
nlp.vocab[5983625672228268878].text
for key, row in nlp.vocab.vectors.key2row.items():
    print(f"{key} => {row}")
    if row > 10:
        break
nlp.vocab[3424551750583975941].text
row2key = {}
for key, row in nlp.vocab.vectors.key2row.items():
    row2key[row] = key
row2key
row2key[0]
row2key.keys()
row2key.keys()[0]
list(row2key.keys())
list(row2key.keys())[0]
sort(list(row2key.keys()))
sorted(list(row2key.keys()))
sorted(list(row2key.keys()))[0]
row2key[1]
nlp.vocab[12646065887601541794].text
nlp.vocab['.'].vector
nlp.vocab.vectors.data[1, :]
nlp.vocab.vectors.data[1, :] == nlp.vocab['.'].vector
diffs.sum(dim=1).argmin()
row2key
row2key[403305]
nlp.vocab[17573681073551504768].text
nlp.vocab[17573681073551504768].text.lower()
%save notebooks/explore-inference-with-spacy
%history -f notebooks/inference_with_spacy.py
