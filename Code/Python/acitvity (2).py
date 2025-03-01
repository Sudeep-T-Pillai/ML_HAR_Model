# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import sklearn
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split

import os
os.makedirs('/content/checkpoint', exist_ok=True)
os.makedirs('/content/saved_model', exist_ok=True)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

sns.set(style='whitegrid', palette= 'muted', font_scale=1.5)

RANDOM_SEED = 42

from google.colab import drive
drive.mount('/content/drive')

columns = ['user','activity','timestamps','x-axis','y-axis','z-axis']
df = pd.read_csv('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header= None, names = columns,on_bad_lines='skip')
df = df.dropna()
df.head()



# Convert columns to strings, remove non-numeric characters, and convert to numeric
df['x-axis'] = pd.to_numeric(df['x-axis'].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
df['y-axis'] = pd.to_numeric(df['y-axis'].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
df['z-axis'] = pd.to_numeric(df['z-axis'].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')

countOfActivities = df['activity'].value_counts()
print(countOfActivities)

N_TIME_STEPS =200
N_FEATURES = 3
step = 20
segments = []
labels = []

for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['x-axis'].iloc[i:i + N_TIME_STEPS].values
    ys = df['y-axis'].iloc[i:i + N_TIME_STEPS].values
    zs = df['z-axis'].iloc[i:i + N_TIME_STEPS].values
    label = df['activity'].iloc[i:i + N_TIME_STEPS].mode()[0]
    segments.append([xs, ys, zs])
    labels.append(label)

np.array(segments).shape

segments = np.asarray(segments, dtype=np.float32)
labels = np.asarray(labels)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1,N_TIME_STEPS,N_FEATURES)
reshaped_segments.shape

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

print(labels)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(
    reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED
)

len(X_train)

len(X_test)

"""BUILDING MODEL"""

N_CLASSES = 6
N_HIDDEN_UNITS = 64

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.compat.v1.Variable(tf.random.normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.compat.v1.Variable(tf.random.normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.compat.v1.Variable(tf.random.normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.compat.v1.Variable(tf.random.normal([N_CLASSES]))
    }

    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    lstm_layers = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_layers)

    outputs, _ = tf.compat.v1.nn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name = 'input')
Y = tf.compat.v1.placeholder(tf.float32, [None, N_CLASSES])

pred_Y = create_LSTM_model(X)
pred_softmax = tf.nn.softmax(pred_Y, name = 'Y_')

l2_loss = 0.0015
l2 = l2_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2

LEARNING_RATE = 0.0025
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Define accuracy metric
correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

N_EPOCHS = 50
BATCH_SIZE = 1024

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.Saver()


history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

for epoch in range(1, N_EPOCHS + 1):
    # Training loop
    for start, end in zip(range(0, len(X_train), BATCH_SIZE), range(BATCH_SIZE, len(X_train) + 1, BATCH_SIZE)):
        _, acc_train, loss_train = sess.run([optimizer, accuracy, loss], feed_dict={
            X: X_train[start:end],
            Y: y_train[start:end]
        })
        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)

    # Testing loop
    acc_test_list = []
    loss_test_list = []
    for start, end in zip(range(0, len(X_test), BATCH_SIZE), range(BATCH_SIZE, len(X_test) + 1, BATCH_SIZE)):
        acc_test, loss_test = sess.run([accuracy, loss], feed_dict={
            X: X_test[start:end],
            Y: y_test[start:end]
        })
        acc_test_list.append(acc_test)
        loss_test_list.append(loss_test)
    acc_test = sum(acc_test_list) / len(acc_test_list)
    loss_test = sum(loss_test_list) / len(loss_test_list)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    # Print epoch results
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Test Accuracy: {acc_test}, Test Loss: {loss_test}')

# Final evaluation on test set
prediction, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={
    X: X_test,
    Y: y_test
})
print(f'Final Test Accuracy: {acc_final}, Final Test Loss: {loss_final}')

pickle.dump(prediction, open("prediction.p", "wb"))
pickle.dump(history, open("history.p","wb"))

tf.io.write_graph(sess.graph_def, '/content/checkpoint', 'HAR.pbtxt')
saver.save(sess, save_path="/content/checkpoint/HAR.ckpt")

history = pickle.load(open("history.p", "rb"))
prediction = pickle.load(open("prediction.p", "rb"))

plt.figure(figsize=(12,8))

plt.plot(np.array(history['train_loss']), "r--", label = "Train loss")
plt.plot(np.array(history['train_acc']), "g--", label = "Train accuracy")

plt.plot(np.array(history['test_loss']), "r--", label = "test loss")
plt.plot(np.array(history['test_acc']), "g--", label = "test accuracy")

plt.title("Training session progress over iteration")
plt.legend(loc = 'upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()

tf.compat.v1.saved_model.simple_save(
    sess,
    "./saved_model/",
    inputs={"X": X},
    outputs={"Y_": pred_softmax}
)

# Convert the model
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("./saved_model/")
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('./saved_model/HAR.tflite', 'wb') as f:
    f.write(tflite_model)

!pip install tflite_support

import json

metadata = {
    "name": "Human Activity Recognition Model",
    "description": "Detects human activities based on accelerometer data",
    "version": "v1",
    "author": "Your Name",
    "license": "Apache License. Version 2.0",
    "input": {
        "name": "input",
        "description": "Accelerometer data",
        "shape": [1, 200, 3],
        "dtype": "float32"
    },
    "output": {
        "name": "output",
        "description": "Activity probabilities",
        "shape": [1, 6],
        "dtype": "float32"
    }
}

with open('./saved_model/HAR_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

sess.close()