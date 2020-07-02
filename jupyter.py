#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SimpleLinearRegression import SimpleLinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('data-base.csv')


# In[3]:


def f_calc(m):
    return m*9.80665
def k_calc(f, m):
    return f/m

def calc_forcas(massas):
    f = []
    for i in massas:
        f.append(f_calc(i))
    return f

def calc_ks(forcas,distancias):
    return np.divide(forcas,distancias)


# In[4]:


data['Força (N)'] = calc_forcas(data['Massa (kg)'])
data['K(N/m)'] = calc_ks(data['Força (N)'],data['Distância (m)'])

data


# In[5]:


X = data['Distância (m)']
Y = data['Força (N)']

datac = data.drop(['Massa (kg)','K(N/m)'], axis=1)

sr = SimpleLinearRegression()
sr.fit(X,Y)

sr.test_model()


# In[6]:


def plot_line(sr,title,label_x,label_y, line_l,value_l):
    x_values = sr.X
    y_values = [sr.predict([x]) for x in x_values]
    
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot(211)
    ax1.set_title(title)
    ax1.set_ylabel(label_y)
    ax1.set_xlabel(label_x)    
    
    line = ax1.plot(x_values,y_values,'r')
    values = ax1.plot(sr.X,sr.y,'bo')
    
    ax1.legend([line_l, value_l])
    
plot_line(sr,'MMQ - Força(N)/distancia(m)','d(m)','F(N)','F = 19.0307 * d + -0.6095','amostragem')


# In[7]:


datac['y: Previsto'] = sr.predicts_
datac['y_r: Real '] = data['K(N/m)'] 

datac['(y- y_r)^2'] = sr.sqe_
datac['(y- y_m)^2'] = sr.sqt_
datac['[(y- y_m)^2 - (y- y_r)^2]'] = sr.sqr_

print("y = ax + b \n")

print("b = " + str(sr.intercept_[0]))
print("a = " + str(sr.coef_[0]))

print("\n")

print("SQT = " + str(sum(sr.sqt_)))
print("SQE = " + str(sum(sr.sqe_)))
print("SQR = " + str(sum(sr.sqr_)))

print("\n")

r2 = sum(sr.sqr_)/sum(sr.sqt_)
print("R = " + str(r2**0.5))
print("Coeficiente de correlação: " + str(r2))

datac

