#!/usr/bin/env python
# coding: utf-8

# # ML Programming Assignment 2
# 
# ## 1. GridSearchCV and Support Vectors
# 
# > 1. Use GridSearchCV to find the best parameters of SVM using the train split.
# Report the train and test accuracy on the best parameters.
# Also state your observations on the best parameter

# In[1]:


import numpy as np
import pandas as pd
import pickle
import feather
import os
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer

#skimage/
from skimage import io
from skimage.transform import resize

#sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.linear_model import SGDClassifier

#skplt
import scikitplot as skplt

#plot
# %matplotlib inline
sns.set_style(style="darkgrid")


# In[2]:


cifar_path = os.getcwd() + '/cifar-10-data/'


# In[3]:


def decoding(d):
    
    datapt = {}
    for k, v in d.items():
        if isinstance(v, bytes):
            datapt[k.decode("utf-8")] = v.decode("utf-8")          
        else:
            datapt[k.decode("utf-8")] = v
            
    return datapt           


# In[4]:


def createDataMatrix(imgs):
    data_mat = []
    data_label = []
    
    for batch in imgs:
        for i in range(0, batch['data'].shape[0]):
            data_mat.append(batch['data'][i, :])
            data_label.append(batch['labels'][i])
    
    labels = np.array(data_label)[:,np.newaxis]
    data_mat = np.append(data_mat, labels, axis=1)
    
    return data_mat


# In[5]:


def collect_data(dataPath):
    every_path = list(os.walk(dataPath))
    train_data = []
    test_data = []
    meta = []
    
    for i in range(0,len(every_path)):
        dirPath, dirName, fileNames = every_path[i]
        folder = dirPath.split('/')[-1]
        
        if(len(dirName) == 0):
            for j in fileNames:
                single_doc_loc = dirPath + '/' + j
                
                with open(single_doc_loc, 'rb') as fo:
                    dt = pickle.load(fo, encoding='bytes')
                    
                    if(folder == 'train'):
                        train_data.append(decoding(dt))
                    elif(folder == 'test'):
                        test_data.append(decoding(dt))
                    elif(folder == 'meta'):
                        meta.append(decoding(dt))
                        
    return train_data, test_data, meta


# In[6]:


train_images, test_images, meta_data =  collect_data(cifar_path)


# In[7]:


train_data = createDataMatrix(train_images)
test_data = createDataMatrix(test_images)


# In[8]:


X_train = train_data[:,:-1]
y_train = train_data[:,-1]

X_test = test_data[:,:-1]
y_test = test_data[:,-1]


# In[28]:


# # Creating dev. set
# X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train,test_size=0.10,
#                                                   stratify=y_train, random_state=42)


# In[74]:


# print(X_train.shape, X_dev.shape, X_test.shape)
print(X_train.shape, X_test.shape)


# In[75]:


# MinMAx Scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


# In[76]:


pca1 = PCA(n_components=50, random_state=0)
get_ipython().run_line_magic('time', 'pca1.fit(X_train[:, :1024])')


# In[77]:


pca2 = PCA(n_components=50, random_state=0)
get_ipython().run_line_magic('time', 'pca2.fit(X_train[:, 1024:2048])')


# In[78]:


pca3 = PCA(n_components=50, random_state=0)
get_ipython().run_line_magic('time', 'pca3.fit(X_train[:, 2048:])')


# In[79]:


X_train = np.hstack((pca1.transform(X_train[:, :1024]), 
                     pca2.transform(X_train[:, 1024:2048]), pca3.transform(X_train[:, 2048:])))


# In[80]:


# X_dev = np.hstack((pca1.transform(X_dev[:, :1024]), 
#                      pca2.transform(X_dev[:, 1024:2048]), pca3.transform(X_dev[:, 2048:])))


# In[81]:


X_test = np.hstack((pca1.transform(X_test[:, :1024]), 
                     pca2.transform(X_test[:, 1024:2048]), pca3.transform(X_test[:, 2048:])))


# In[82]:


# plt.plot(list(range(1,3073)), np.cumsum(pca.explained_variance_ratio_))
# np.cumsum(pca.explained_variance_ratio_)[100]
# plt.show()


# In[83]:


print(X_train.shape, X_test.shape)


# In[61]:


# np.cumsum(pca.explained_variance_ratio_)[500]


# > 1

# In[39]:


# def cal_f1(y, y_pred):
#     return np.sqrt(f1_score(y, y_pred, average='micro'))

# f1 = make_scorer(cal_f1, greater_is_better=True)


# In[ ]:


params = {
    'kernel':['rbf', 'linear'],
    'C':[1e-4, 1e-3, 1e-2, 0.1, 1, 2, 4, 10, 100],
#     'gamma':['scale', 'auto']
}

svc = SVC(random_state=0)
clf = GridSearchCV(svc, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3, verbose=3)

start = timer()
# %time clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
end = timer()
print('Training time : '+str(end - start)+'s')


# In[64]:


# pd.DataFrame(clf.cv_results_)


# In[41]:


print('#### Acc Score ####')
start = timer()
print('Train:', clf.score(X_train, y_train))
end = timer()
print(end-start)

start = timer()
# print('Dev :', clf.score(X_dev, y_dev))
print('Test:', clf.score(X_test, y_test))
end = timer()
print(end-start)


# In[ ]:


# train_pred = clf.predict(X_train)
# dev_pred = clf.predict(X_dev)
# test_pred = clf.predict(X_test)


# In[42]:


print('No. of support vectors: ', clf.support_.shape)


# In[43]:


new_training_set = X_train[clf.support_,:]
new_training_label = y_train[clf.support_]


# In[44]:


clf2 = SVC(kernel='linear', C=0.01, random_state=0)
# clf = GridSearchCV(svc, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3, verbose=3)

start = timer()
# %time clf.fit(X_train, y_train)
clf2.fit(new_training_set, new_training_label)
end = timer()
print('Training time : '+str(end - start)+'s')


# In[45]:


start = timer()
print('Accuracy on old Training set:', clf2.score(X_train, y_train))
end = timer()
print(end-start)

start = timer()
print('Test:', clf2.score(X_test, y_test))
end = timer()
print(end-start)


# In[98]:


# skplt.metrics.plot_roc(y_test, proba, plot_macro=False, plot_micro=False, figsize=(8,6))
# plt.show()


# In[ ]:


# skplt.metrics.plot_confusion_matrix(y_test, test_pred, normalize=False)
# plt.plot()


# In[47]:


# print(clf2.support_.shape)


# In[ ]:




