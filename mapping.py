#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math
import matplotlib as plt
import matplotlib.pyplot as plt
data = pd.read_csv('examp3.txt', delimiter=',|;', names = []).to_numpy()
data


# In[38]:


angle = np.arange(-120, 120, 240/681)
res = np.arange(0)
x_lid = np.arange(0)
y_lid = np.arange(0)
for j in range(100):
    x_lid = np.append(x_lid, 0.3*math.cos(data[j][2])+data[j][0])
    y_lid = np.append(y_lid, 0.3*math.sin(data[j][2])+data[j][1])

plt.scatter(x_lid, y_lid)


# In[40]:


x_rob = np.arange(0)
y_rob = np.arange(0)
for j in range(100):
    x_rob = np.append(x_rob, data[j][0])
    y_rob = np.append(y_rob, data[j][1])

plt.scatter(x_lid, y_lid)
plt.scatter(x_rob, y_rob, c = 'deeppink')


# In[27]:


#for j in range(100):
#j = 0
x = np.arange(0)
y = np.arange(0)
for j in range(100):
    for i in range(681):
        if (2 < data[j][i+3] < 5.6):
            x = np.append(x, x_lid[j] + data[j][i+3]*math.cos(math.radians(-angle[i]) + data[j][2]))
            y = np.append(y, y_lid[j] + data[j][i+3]*math.sin(math.radians(-angle[i]) + data[j][2]))
    
    #res = np.append(res, [[data[j][i+3]*math.cos(math.radians(angle[i])), data[j][i+3]*math.sin(math.radians(angle[i]))]])

#type(res)
#res = res.reshape(681, 2)
#print(res)
plt.scatter(x, y, 1)
#plt.scatter(x_lid, y_lid)
plt.scatter(x_rob, y_rob, 1, c = 'deeppink')


# In[9]:


#for j in range(100):
j = 0
x_1 = np.arange(0)
y_1 = np.arange(0)
#for j in range(5):
for i in range(681):
    if (data[j][i+3] < 5.6):
        x_1 = np.append(x_1, x_lid[j] + data[j][i+3]*math.cos(math.radians(angle[i]) + data[j][2]))
        y_1 = np.append(y_1, y_lid[j] + data[j][i+3]*math.sin(math.radians(angle[i]) + data[j][2]))
    
    #res = np.append(res, [[data[j][i+3]*math.cos(math.radians(angle[i])), data[j][i+3]*math.sin(math.radians(angle[i]))]])

#type(res)
#res = res.reshape(681, 2)
#print(res)
plt.scatter(x_1, y_1)
#for j in range(100):
j = 1
x_2 = np.arange(0)
y_2 = np.arange(0)
#for j in range(5):
for i in range(681):
    if (data[j][i+3] < 5.6):
        x_2 = np.append(x_2, x_lid[j] + data[j][i+3]*math.cos(math.radians(angle[i]) + data[j][2]))
        y_2 = np.append(y_2, y_lid[j] + data[j][i+3]*math.sin(math.radians(angle[i]) + data[j][2]))
    
    #res = np.append(res, [[data[j][i+3]*math.cos(math.radians(angle[i])), data[j][i+3]*math.sin(math.radians(angle[i]))]])

#type(res)
#res = res.reshape(681, 2)
#print(res)
plt.scatter(x_2, y_2)


# In[81]:


print(data[0][0])
type(data)


# In[53]:


result = data[0][0]*angle[0]


# In[56]:


result = np.arange(0)
result


# In[83]:


res


# In[156]:


pip install matplotlib


# In[98]:


pip install simplification


# In[47]:


from simplification.cutil import simplify_coords
massivik = np.vstack([x, y])
massivik = np.transpose(massivik)
simplified = simplify_coords(massivik, 0.01)
simplified.shape
sim_x = np.arange(0)
sim_y = np.arange(0)
simplified = np.transpose(simplified)
for i in range(3581):
    sim_x = np.append(sim_x, simplified[0][i])
    sim_y = np.append(sim_y, simplified[1][i])
plt.scatter(sim_x, sim_y, 1)
#plt.scatter(x_lid, y_lid)
plt.scatter(x_rob, y_rob, 1, c = 'deeppink')


# In[46]:


simplified.shape


# In[121]:


massivik = np.vstack([x, y])
massivik = np.transpose(massivik)
massivik


# In[ ]:


simplified = simplify_coords(coords, 1.0)


# In[146]:


angle = np.arange(-120, 120, 240/681)


# In[147]:


pip install rdp


# In[30]:


from rdp import rdp
mask = rdp(massivik, epsilon = 0.01, algo = "iter" ,  return_mask = True )


# In[31]:


mask


# In[15]:


kk = massivik[mask]
kk.shape


# In[32]:


sim_x = np.arange(0)
sim_y = np.arange(0)
kk = np.transpose(kk)
for i in range(6045):
    sim_x = np.append(sim_x, kk[0][i])
    sim_y = np.append(sim_y, kk[1][i])
plt.scatter(sim_x, sim_y, 1)
plt.scatter(x_rob, y_rob, 1, c = 'deeppink')


# In[ ]:




