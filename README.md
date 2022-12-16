# CLTK models for Middle English

## How the word2vec model was trained

We first need to install some mathematics and machine learning packages.

```bash
pip install scipy sklearn gensim
```


We import a corpus of Middle English texts
```python
import pickle
with open('./middle_english_txt-.p','rb') as p:
    corpus = pickle.load(p)
```

The normalization process involves removal of punctuation, special characters and numbers.

```python
CHARACTERS_TO_REMOVE = "..."

def remove_punc(text):
    for ele in text:
        if ele in CHARACTERS_TO_REMOVE:
            return text.lower().replace(ele, "")
        else:
            return text.lower()
            
new_data = []

for texts in corpus:
    part_text = []
    for word in texts:
        part_text.append(remove_punc(word))
    new_data.append(part_text)
```

The data was in a form of a list of lists of strings or a list of sentences, where a sentence is a list of words.

Then we use `Word2Vec` class from **gensim**
```python
# time to try gensim to create word2vecs
# see NLP in action, 6.2.4
from gensim.models.word2vec import Word2Vec

num_features = 50
min_word_count = 30
num_workers = 2
window_size = 20
subsampling = 1e-3

model = Word2Vec(
    new_data,
    workers=num_workers,
    vector_size=num_features,
    min_count=min_word_count,
    window=window_size,
    sample=subsampling)

model.init_sims(replace=True)
me_w2v_model = "me_word_embeddings_model.bin"
model.save(me_w2v_model)

```

The model is now saved in the file `me_word_embeddings_model.bin`.