#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing tensorflow library
import tensorflow as tf
#importing keras module form tensorflow library
from tensorflow import keras
#importing squential class from keras module in tensorflow
from tensorflow.keras  import Sequential
#importing layers class from keras module in tensorflow
from tensorflow.keras import layers
# Importing the Dense and Flatten layers from the keras.layers module in TensorFlow
from tensorflow.keras.layers import Dense,Flatten
# Importing the mnist dataset from the keras.datasets module in TensorFlow
from tensorflow.keras.datasets import mnist


# In[2]:


#loading the dataset in test,train 
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[3]:


#finding shape of x_trian
x_train.shape


# In[4]:


# Normalizing the training data by scaling the pixel values to the range [0, 1]
x_train=x_train/255.0


# In[5]:


x_train[0]


# In[10]:


# Creating an instance of the Sequential model
model = Sequential()


# In[18]:


#converting mutidimension to single dimension
model.add(Flatten(input_shape=(28,28)))


# In[19]:


# Adding a Dense layer with 128 units and ReLU activation function
model.add(Dense(128, activation='relu'))


# In[20]:


model.add(Dense(10,activation='softmax'))


# In[21]:


#model.compile(loss='spare_categorical_crossentropy',optimizer='Adam')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


# In[22]:


model.fit(x_train,y_train,epochs=3,validation_split=0.2)


# In[23]:


model.predict(x_test[0].reshape(1,28,28)).argmax(axis=-1)


# In[24]:


x_test=x_test/255.0


# In[25]:


y_prediction=model.predict(x_test)


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


import numpy as np


# In[64]:


from sklearn.metrics import accuracy_score

# Assuming `predictions` contains the model's output probabilities
# Convert probabilities to class labels
y_pred_labels = np.argmax(y_prediction, axis=1)

# Calculate the accuracy score
accuracy_score_ann = accuracy_score(y_test, y_pred_labels)


# In[69]:


accuracy_score_ann*100


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


plt.imshow(x_test[0])


# In[ ]:




