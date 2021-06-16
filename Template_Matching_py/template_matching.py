#!/usr/bin/env python
# coding: utf-8

# ## Importing Modules

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np



# ## Reading Image

# In[2]:


img_bgr = cv2.imread('coins.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ## Displaying images in different colour forms

# In[3]:


img_box = [img_bgr, img_rgb, img_gray]

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 20))

for i in range(0, 3):
    ax = axs[i]
    ax.imshow(img_box[i])

plt.show()


# ## Reading Image Template

# In[4]:


img_template = cv2.imread('coin_template.png')
img_template_rgb = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB)
img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)


# ## Displaying Image Template in different colour forms

# In[5]:


img_template_box = [img_template, img_template_rgb, img_template_gray]

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (5, 5))

for i in range(0, 3):
    ax = axs[i]
    ax.imshow(img_template_box[i])

plt.show()


# In[6]:


img_template_gray.shape


# ## Matching-Template

# In[7]:


result = cv2.matchTemplate(img_gray, img_template_gray, cv2.TM_CCOEFF_NORMED)

# 0.8 is used for 80%
threshold = 0.8

loc = np.where(result >= threshold)


# ## Obtaining height and width of template image

# In[8]:


h, w = img_template_gray.shape[0], img_template_gray.shape[1]


# In[9]:


img_rgb.shape


# ## Plotting rectangles in the original image

# In[10]:


for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2)
    cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


# ## Displaying Processed Image

# In[11]:


plt.imshow(img_rgb)


# ## Saving Processed Image

# In[12]:


cv2.imwrite('processed_image.png', img_bgr)


# In[ ]:




