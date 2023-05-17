#adapted from DeepfakeTextDetection

import tensorflow as tf
import numpy as np
import pandas as pd
import functools
import os
from sklearn import preprocessing as prep
from sklearn.metrics import classification_report


def add_1(x):
    return x+1

def overX(x):
    return 1/x


our_data_file = "./gltr_processed.jsonl"

df = pd.read_json(our_data_file, lines=True)
df.head()


#encode labels
df = df.groupby("label").head(min(df["label"].value_counts()))
df = df.sample(frac=1)
labels = df.pop("label")
le = prep.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)

#load data into matrix
size = df.size
features = np.zeros((df.size, 250))
count = 0
payloads = df.pop("Payload")
for entry in payloads:
    splits = entry.split('-')
    length = len(splits)
    while length < 250:
        splits.append(0)
        length = length + 1
    splits = splits[0:250]
    features[count, :] = np.asarray(splits)
    count = count + 1

    

trainFeats = features
trainLabels = np.asarray(labels)


#normalize data
add1 = np.vectorize(add_1)
overX = np.vectorize(overX)
trainFeats = add1(trainFeats)
trainFeats = overX(trainFeats)


#train test split
data_size = size
splitPoint =(data_size*4)//5
testFeats = trainFeats[splitPoint:, :]
trainFeats=trainFeats[:splitPoint, :]
testLabels = trainLabels[splitPoint:]
trainLabels = trainLabels[:splitPoint]



#model compilation and training
model = tf.keras.models.Sequential([tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dense(250, activation='tanh'),
                                    tf.keras.layers.Dropout(.01),
                                    tf.keras.layers.Dense(2, activation='softmax')])
optimizer = tf.keras.optimizers.Adam(learning_rate = .0001)







model.compile(optimizer=optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics= [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)])

model.fit(x = trainFeats, y = trainLabels, epochs = 10, batch_size = 10, shuffle = False)

results = model.evaluate(x = testFeats, y = testLabels, batch_size = 1)

#show results
print(results)
y_pred = model.predict(testFeats, batch_size =1, verbose=1)
y_pred_bool = np.argmax(y_pred, axis = 1)
print(classification_report(testLabels, y_pred_bool))
print(testLabels)
print(y_pred_bool)

#save model

model.save('./my_model')


