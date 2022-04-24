# Toy semi-supervised NER
-------------------------


Here I experiment just for fun, creating semi-supervised method to perform training of NER model without labeled training data available - an approach, which can be used for training NER classifiers for specific text corpora. All that is needed is as much as possible examples of named entities which will occur in those data.<br>
The goal is to deal with the problem of the lack of supervised data for training NER models. Two most known datasets are CONLL2003 (~4 MB) and WNUT-17 (2.3 mb) [1](https://github.com/davidsbatista/NER-datasets). Also, which is more important, it is almost impossible to find supervised training data for specific domain (medical, twitter, etc). Using Amazon Mechanical Turk authors of the paper Annotating Large Email Datasets for Named Entity Recognition with Mechanical Turk [2](http://www.aclweb.org/anthology/W10-0712) managed to collect 8mb dataset after 4 months, thus creating own dataset is long and expensive.<br>
The main assumption behind the usage of such small datasets is that word embeddings should capture contextual information, so the features indicating particular word being an enitity should be somehow encoded in the embeddings. Thus training model on a small dataset is just a sort of a training new simple classifier over existing embeddings and should not require much data. However, perhaps existing 3-8mb datasets are insufiicient for capturing all the information. <br>

The idea of this toy model is the following: all the named entities of some type (`Company` only at the moment) should have something in common in the context where they appear, so even the one who does not know such a company, may deduce that this is a company name in text. Thus, by providing lists of companies, persons, etc I try to create context embeddings using surrounding words in the chosen corpora, and train classifier for them. After that, another named entity, which did not occur during training, should be correctly classified by its context in this text domain. <br>

## Data and model

Training data is scraped from Wikipedia using the list of companies; article about each company is scanned, sentences containing company name are split into two parts which are fed into two biLSTMs. Characted embeddings are used for analyzed word or phrase and fed to third biLSTM. Summarizing layer is set above all three LSTMs.

To feed data into the model, I use GoogleNews word2vec embeddings [3](https://github.com/mmihaltz/word2vec-GoogleNews-vectors). Zero embeddings are used for unknown words. Please edit the code in `train.py` and set path. Negative examples are created by splitting the sentences, which does not contain company name, at random word. The resulting training dataset is about 11 MB.

## Requirements

Python 3, Keras framework.


## Output

There is no need to export toy model in `train.py`, as it creates limited vocabulary matrix from the dataset. The quality is of course **very** low, but this idea of context capturing can be used as a part for something bigger. You can train it in the python shell and play a bit by evaluating different sentences:

`$python3 -i train.py`

```
Loading word2vec model
Loading datafiles
positive examples: 17503
negative examples: 56105
Building vocabulary
Constructing embedding matrix
Preparing model inputs
vocab_size: 57393
max_len: 15

Train on 66247 samples, validate on 7361 samples
Epoch 1/7
66247/66247 [==============================] - 211s 3ms/step - loss: 0.2955 - acc: 0.8771 - val_loss: 0.2347 - val_acc: 0.8995

Epoch 00001: val_loss improved from inf to 0.23466, saving model to models/model-best.hdf5
Epoch 2/7
66247/66247 [==============================] - 216s 3ms/step - loss: 0.1573 - acc: 0.9373 - val_loss: 0.1176 - val_acc: 0.9554

Epoch 00002: val_loss improved from 0.23466 to 0.11764, saving model to models/model-best.hdf5
Epoch 3/7
66247/66247 [==============================] - 218s 3ms/step - loss: 0.0881 - acc: 0.9676 - val_loss: 0.0830 - val_acc: 0.9686

Epoch 00003: val_loss improved from 0.11764 to 0.08302, saving model to models/model-best.hdf5
Epoch 4/7
66247/66247 [==============================] - 219s 3ms/step - loss: 0.0576 - acc: 0.9802 - val_loss: 0.0696 - val_acc: 0.9761

Epoch 00004: val_loss improved from 0.08302 to 0.06955, saving model to models/model-best.hdf5
Epoch 5/7
66247/66247 [==============================] - 219s 3ms/step - loss: 0.0438 - acc: 0.9851 - val_loss: 0.0577 - val_acc: 0.9803

Epoch 00005: val_loss improved from 0.06955 to 0.05767, saving model to models/model-best.hdf5
Epoch 6/7
66247/66247 [==============================] - 219s 3ms/step - loss: 0.0347 - acc: 0.9883 - val_loss: 0.0651 - val_acc: 0.9787

Epoch 00006: val_loss did not improve from 0.05767
Epoch 7/7
66247/66247 [==============================] - 218s 3ms/step - loss: 0.0303 - acc: 0.9900 - val_loss: 0.0631 - val_acc: 0.9803

Epoch 00007: val_loss did not improve from 0.05767


>>> ner(['thecorp', 'sued', 'us', 'for','million','dollars'])
thecorp(True)
sued(False)
us(False)
for(False)
million(False)
dollars(False)


>>> ner(['we','think','msnbc','is','responsible'])
we(False)
think(False)
msnbc(True)
is(False)
responsible(False)

>>> ner(['thecorp', 'formed', 'in', '1997','from','merger'])
thecorp(True)
formed(False)
in(False)
1997(False)
from(False)
merger(False)
```

Perhaps I will try to train on much bigger corpora later.
