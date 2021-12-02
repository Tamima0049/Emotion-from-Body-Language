#!/usr/bin/env python
# coding: utf-8

# In[13]:


from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import io
import os
from PIL import Image


import numpy as np
from skimage import io
import os
from PIL import Image


datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.2,1.0],
        #horizontal_flip=True,
        fill_mode='nearest')#, cval=125)    #Also try nearest, constant, reflect, wrap


image_directory = '/Users/tamima_rashid/Desktop/Data/IRB/IRB_CNN/IRB_Fear/'
#SIZE = 128
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'png'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        #image = image.resize((261,319))
        #image = image.crop((0,0,640,325))
        dataset.append(np.array(image))

x = np.array(dataset)

#Let us save images to get a feel for the augmented images.
#Create an iterator either by using image dataset in memory (using flow() function)
#or by using image dataset from a directory (using flow_from_directory)
#from directory can beuseful if subdirectories are organized by class
   
# Generating and saving 10 augmented samples  
# using the above defined parameters.  
#Again, flow generates batches of randomly augmented images
  
i = 0
for batch in datagen.flow(x, batch_size=125,  
                          save_to_dir='/Users/tamima_rashid/Desktop/Data/IRB/IRB_CNN/IRB_Fear_aug/', 
                          save_prefix='augbatch1', 
                          save_format='png'):
    i += 1
    if i > 0:
        break  # otherwise the generator would loop indefinitely  


# In[ ]:





# In[ ]:




