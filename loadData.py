# coding: utf-8
# In[1]:
import turicreate as tc

# In[2]:
# Load images (Note: you can ignore 'Not a JPEG file' errors)
data = tc.image_analysis.load_images('PetImages', with_path=True)

# In[3]:
# From the path-name, create a label column
data['label'] = data['path'].apply(lambda path: 'dog' if '/Dog' in path else 'cat')

# In[4]:
# Save the data for future use
data.save('cats-dogs.sframe')

# In[5]:
# Explore interactively
data.explore()

