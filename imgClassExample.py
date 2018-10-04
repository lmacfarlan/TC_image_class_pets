
# coding: utf-8

# In[1]:


import turicreate as tc


# In[2]:


# Load the data
data =  tc.SFrame('cats-dogs.sframe')


# In[3]:


# Make a train-test split
train_data, test_data = data.random_split(0.8)


# In[4]:


# Create the model
model = tc.image_classifier.create(train_data, target='label')


# In[ ]:


# Save predictions to an SArray
predictions = model.predict(test_data)


# In[ ]:


# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test_data)
print(metrics['accuracy'])


# In[ ]:


# Save the model for later use in Turi Create
model.save('mymodel.model')

