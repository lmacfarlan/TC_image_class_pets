# coding: utf-8
# In[1]:
import turicreate as tc
from s3fs.core import S3FileSystem
from datetime import datetime, timedelta
import os
import uuid
from common.logger import get_logger

# Set constants
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# In[2]:
# Load the data
data =  tc.SFrame('cats-dogs.sframe')

# In[3]:
# Make a train-test split
train_data, test_data = data.random_split(0.8)

# In[4]:
# Create the model
model = tc.image_classifier.create(train_data, target='label')

# In[5]:
# Save predictions to an SArray
predictions = model.predict(test_data)

# In[6]:
# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test_data)
print(metrics['accuracy'])

# In[7]:
# Save the model for later use in Turi Create
model.save('mymodel.model')

