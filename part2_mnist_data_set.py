#!/usr/bin/env python
# coding: utf-8

# In[2]:


# python notebook for Make Your Own Neural Network
# working with the MNIST data set
#
# (c) Tariq Rashid, 2016
# license is GPLv2


# In[3]:


import numpy
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# open the CSV file and read its contents into a list
data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()


# In[5]:


# check the number of data records (examples)
len(data_list)


# In[6]:


# show a dataset record
# the first number is the label, the rest are pixel colour values (greyscale 0-255)
data_list[1]


# In[7]:


# take the data from a record, rearrange it into a 28*28 array and plot it as an image
all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')


# In[8]:


# scale input to range 0.01 to 1.00
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)


# In[12]:


#output nodes is 10 (example)
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99


# In[13]:


print(targets)


# In[ ]:




