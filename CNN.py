#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array


# In[19]:


TRAIN_DIR = 'C:/Users/msi/CNN/trainSet/'
train_folder_list = array(os.listdir(TRAIN_DIR))
 
train_input = []
train_label = []
 
label_encoder = LabelEncoder()  # LabelEncoder Class 호출
integer_encoded = label_encoder.fit_transform(train_folder_list)
onehot_encoder = OneHotEncoder(sparse=False) 
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
 
for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        train_input.append([np.array(img)])
        train_label.append([np.array(onehot_encoded[index])])
 
train_input = np.reshape(train_input, (-1, 784))
train_label = np.reshape(train_label, (-1, 10))
train_input = np.array(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)
np.save("train_data.npy", train_input)
np.save("train_label.npy", train_label)


# In[20]:


TEST_DIR = 'C:/Users/msi/CNN/testSet'
test_folder_list = array(os.listdir(TEST_DIR))
 
test_input = []
test_label = []
 
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_folder_list)
 
onehot_encoder = OneHotEncoder(sparse=False) 
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
 
for index in range(len(test_folder_list)):
    path = os.path.join(TEST_DIR, test_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_input.append([np.array(img)])
        test_label.append([np.array(onehot_encoded[index])])

test_input = np.reshape(test_input, (-1, 784))
test_label = np.reshape(test_label, (-1, 3))
test_input = np.array(test_input).astype(np.float32)
test_label = np.array(test_label).astype(np.float32)
np.save("test_input.npy",test_input)
np.save("test_label.npy",test_label)


# In[ ]:


import tensorflow as tf
from tqdm import tqdm_notebook
import time

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X,[-1,28,28,1])
Y = tf.placeholder(tf.float32,[None,3])

learning_rate = 0.001
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    #    Conv     -> (?, 28, 28, 32)
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    #    Pool     -> (?, 14, 14, 32)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    #  L1 ImgIn shape=(?, 14,14, 3)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    #    Conv      ->(?, 14, 14, 64)
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    #    Pool      ->(?, 7, 7, 64)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    # L2 ImgIn shape=(?, 7, 7, 64)

with tf.name_scope("Layer3"):
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    #    Conv      ->(?, 7, 7, 128)
    #    Pool      ->(?, 4, 4, 128)
    #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
    # L3 ImgIn shape=(?, 128 * 4 * 4)

W4 = tf.get_variable("F.C", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
# L4 FC 4x4x128 inputs -> 625 outputs


W5 = tf.get_variable("Final.F.C", shape=[625, 3],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([3]))
logits = tf.matmul(L4, W5) + b5
# W5 Final FC 625 inputs -> 10 outputs


# In[ ]:


with tf.name_scope("Cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    tf.summary.scalar("Cost", cost)
    
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)


# In[ ]:


print('Learning started! Please Waite.')

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# tensorboard저장 및 실행
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("C:/Users/msi/CNN/graph")
writer.add_graph(sess.graph)  # Show the graph

    
train_input_1=train_input.reshape(len(train_input),784)
test_input_1=test_input.reshape(len(test_input),784)
for step in tqdm_notebook(range(1000)):
    
    acc_val, summary, cost_val, _= sess.run([accuracy, merged_summary, cost, optimizer], 
                        feed_dict = {X: train_input_1, Y: train_label, keep_prob : 0.7})
        
    writer.add_summary(summary, global_step=step)
        
    if step % 100 == 0 or step==1000:
        print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))
        time.sleep(0.1)
print('Learning Finished!')


# In[ ]:


import random
print('Accuracy:', sess.run(accuracy, feed_dict={
        X: test_input, Y: test_label, keep_prob: 1}))
    
r = random.randint(0, len(test_input) - 1)
print("Label: ", sess.run(tf.argmax(test_label[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: test_input_1[r:r + 1], keep_prob : 1}))


# In[ ]:




