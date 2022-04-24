from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Dot, Concatenate, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
import gensim
import random

MODEL_DIR = "models"
WORD2VEC_BIN =  os.getcwd() + "/GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_EMBED_SIZE = 300

POSITIVE_EXAMPLES = 'positive.txt'
NEGATIVE_EXAMPLES = 'negative.txt'


MAX_LEN = 15 # max amount of surrounding words

EMBED_SIZE = 64 # parameter to qickly change many sizes in network
BATCH_SIZE = 32
NBR_EPOCHS = 7


print('Loading word2vec model')
model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_BIN, binary=True)

NULLVECT = np.array([0]*WORD2VEC_EMBED_SIZE)
def get_vector(word):
    if not word:
        return NULLVECT
    try:
        return model[word]
    except:
        return NULLVECT
    
print("Creating character embeddings")
# Create character embeddings by averaging word embeddings which contain this character (https://github.com/minimaxir/char-embeddings)
char_vectors = {}
# average char embeddings
for word in model.key_to_index:
    vec = model[word]
    for char in word:
        if ord(char) < 128:
            if char in char_vectors:
                char_vectors[char] = (char_vectors[char][0] + vec,
                                 char_vectors[char][1] + 1)
            else:
                char_vectors[char] = (vec, 1)
# averaging
for c, data in char_vectors.items():
    char_vectors[c] = data[0]/data[1]
    
def get_char_vector(c):
    if not c:
        return NULLVECT
    try:
        return char_vectors[c]
    except:
        return NULLVECT


print('Loading datafiles')

# load data, should be lowercase
data = []
entities = []

num_pos = 0
with open("positive.txt", "r") as f:
    for line in f:
        sline = line.strip().split('\t')
        if len(sline)==3: # word (not needed), left and right parts of the sentence
            data.append([1, sline[1].split(' '), sline[2].split(' '), sline[0]])
            entities.append(sline[0])
            num_pos += 1
            
print('positive examples: {}'.format(num_pos))
    
num_neg = 0
with open("negative.txt", "r") as f:
    for line in f:
        sline = line.strip().split('\t')
        if len(sline)==3: # word (not needed), left and right parts of the sentence
            data.append([0, sline[1].split(' '), sline[2].split(' '), sline[0]])
            entities.append(sline[0])
            num_neg += 1
print('negative examples: {}'.format(num_neg))
            

# truncate buggy inputs
for d in data:
    if len(d[1])> MAX_LEN: 
        d[1] = d[1][-MAX_LEN:]
    if len(d[2])> MAX_LEN: 
        d[2] = d[2][:MAX_LEN]


# deterimne max sequence len for LSTM input
left_maxlen = max([len(d[1]) for d in data])
right_maxlen = max([len(d[2]) for d in data])
seq_maxlen = max([left_maxlen, right_maxlen])
entity_maxlen = len(max(entities, key=len))


print('Building vocabulary')
    
    
# build vocabulary
vocab = [''] # trick to have zero-padding return nullvects
for d in data:
    vocab += d[1]
    vocab += d[2]
vocab = {w:idx for idx, w in enumerate(set(vocab))}

vocab_size = len(vocab)


char_vocab = {'':0}
for idx, c in enumerate(char_vectors.keys()):
    char_vocab[c] = idx+1

print('Constructing embedding matrix')

embedding_matrix = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
for word, idx in vocab.items():
    embedding_matrix[idx] = get_vector(word)
    
entity_embedding_matrix = np.zeros((len(char_vocab), WORD2VEC_EMBED_SIZE))
for c, idx in char_vocab.items():
    entity_embedding_matrix[idx] = get_char_vector(c)



print('Preparing model inputs')
# random shuffle data
perm = list(range(len(data)))
random.shuffle(perm)
data = [data[index] for index in perm]

# create input sequences
inputs_left = []
inputs_right = []
inputs_entity = []
outputs = []

for d in data:
    if d[0] == 0:
        outputs.append(np.array([0,1]))
    else:
        outputs.append(np.array([1,0]))
    # fast index
    inputs_left.append([vocab.get(w,0) for w in d[1]])
    inputs_right.append([vocab.get(w,0) for w in d[2]])
    inputs_entity.append([char_vocab.get(c,0) for c in d[3]])

#inputs_left = np.array(inputs_left)
#inputs_right = np.array(inputs_right)
outputs = np.array(outputs)
    
inputs_left = pad_sequences(inputs_left, maxlen=seq_maxlen)
inputs_right = pad_sequences(inputs_right, maxlen=seq_maxlen)
inputs_entity = pad_sequences(inputs_entity, maxlen=entity_maxlen)

inputs_left = [np.array(x).ravel() for x in inputs_left]
inputs_right = [np.array(x).ravel() for x in inputs_right]
inputs_entity = [np.array(x).ravel() for x in inputs_entity]


print('vocab_size: {}'.format(vocab_size))
print('max_len: {}'.format(seq_maxlen))
print('max_entity_len: {}'.format(entity_maxlen))



# left part
left_enc = Sequential()
left_enc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen, weights=[embedding_matrix], trainable=False))

left_enc.add(Dropout(0.2))

left_enc.add(Bidirectional(LSTM(EMBED_SIZE*2, return_sequences=True), 
                       merge_mode="sum"))

left_enc.add(Convolution1D(EMBED_SIZE, 3, padding="valid",activation='relu'))
left_enc.add(MaxPooling1D(pool_size=2, padding="valid"))

# right part
right_enc = Sequential()
right_enc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen, weights=[embedding_matrix], trainable=False))

right_enc.add(Dropout(0.2))

right_enc.add(Bidirectional(LSTM(EMBED_SIZE*2, return_sequences=True),
                       merge_mode="sum"))
right_enc.add(Convolution1D(EMBED_SIZE, 3, padding="valid",activation='relu'))
right_enc.add(MaxPooling1D(pool_size=2, padding="valid"))


# Read entity characters
entity_enc = Sequential()
entity_enc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=len(char_vocab.keys()),
                   input_length=entity_maxlen, weights=[entity_embedding_matrix], trainable=False))

entity_enc.add(Bidirectional(LSTM(EMBED_SIZE*2, return_sequences=True), 
                       merge_mode="sum"))

entity_enc.add(Convolution1D(EMBED_SIZE, 3, padding="valid",activation='relu'))
entity_enc.add(MaxPooling1D(pool_size=2, padding="valid"))


# summarizing
# we don't care about axis, we will flatten anyway
attOut = Concatenate(axis=-2)([left_enc.output, right_enc.output, entity_enc.output]) 
attOut = Flatten()(attOut) #shape is now only (samples,)
attOut = Dense(EMBED_SIZE // 2,activation='tanh')(attOut)

Out = Dense((2),activation='softmax')(attOut)

model = Model([left_enc.input,right_enc.input, entity_enc.input],Out)

model.compile(optimizer="nadam", loss="binary_crossentropy",
metrics=["accuracy"])

model.summary()

checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "model-best.hdf5"),
    verbose=1, save_best_only=True)

model.fit([inputs_left, inputs_right, inputs_entity], outputs, batch_size=BATCH_SIZE,epochs=NBR_EPOCHS, validation_split=0.1,callbacks=[checkpoint], shuffle=True)



# testing -----------------------

def ner(sentence):    
    predictions = []        
    for idx in range(len(sentence)):
        # prepare data
        inputs_left = [vocab.get(w,0) for w in sentence[:idx]]
        inputs_right = [vocab.get(w,0) for w in sentence[min(idx+1,len(sentence)-1):]]   
        inputs_entity = [char_vocab.get(c,0) for c in sentence[idx]]
        inputs_left = pad_sequences([inputs_left], maxlen=seq_maxlen)[0].ravel()
        inputs_right = pad_sequences([inputs_right], maxlen=seq_maxlen)[0].ravel()   
        inputs_entity = pad_sequences([inputs_entity], maxlen=entity_maxlen)[0].ravel()   
        
        prediction = model.predict([inputs_left[np.newaxis,:], inputs_right[np.newaxis,:], inputs_entity[np.newaxis,:]])
        predictions.append(prediction[0])     
        print('{}({})'.format(sentence[idx], prediction[0][0]>0.5))
    return predictions
        
ner(['we','sued','thecorp','for','million','dollars'])

ner(['thecorp', 'formed', 'in', '1997','from','merger'])


