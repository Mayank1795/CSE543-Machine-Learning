#!/usr/bin/env python
# coding: utf-8

# # ML Programming Assignment 2
# 
# ## 2. SVM vs Naive Bayes vs Decision Trees
# 
# > Use the Sklearn’s wine dataset and perform the following tasks. Use train test split
# (Scikit-learn’s) to make a stratified split of 70-30 train-test with seed 42.

# In[288]:


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
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#skplt
import scikitplot as skplt

#plot
# %matplotlib inline
sns.set_style(style="darkgrid")


# In[289]:


data = load_wine() # data['target_names']
print('Data shape : ', data['data'].shape)
print('Features names :', data['feature_names'])


# In[290]:


complete_data = data['data']


# In[291]:


# Creating dev. set
X_train, X_test, y_train, y_test = train_test_split(complete_data, data['target'], test_size=0.30,
                                                  stratify=data['target'], random_state=42)


# In[292]:


print('Training set: ', X_train.shape)
print('Test set: ', X_test.shape)


# In[223]:


# MinMAx Scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Plot pairwise relations in dataset using seaborn and give inferences

# In[119]:


print(np.unique(y_train, return_counts=True))


# In[41]:


df = pd.DataFrame(X_train, columns=data['feature_names'])
df_corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True) # parameter values used from seaborn docs.
sns.heatmap(df_corr, square=True, cmap=cmap, linewidths=0.5)


# In[ ]:


df['class'] = y_train
pair_plot = sns.pairplot(df, vars=df.columns[:-1], hue='class', palette="husl", markers=["o", "s", "D"])
pair_plot.savefig('pairs_plot.png')


# ## Implement One-vs-One LinearSVM

# In[224]:


# 01 (positive class is 1)
pos_one = np.where(np.isin(y_train, [0, 1]))[0]
X_train1 = X_train[pos_one, :]
y_train1 = y_train[pos_one]

# 12 (positive class is 2)
pos_two = np.where(np.isin(y_train, [1, 2]))[0]
X_train2 = X_train[pos_two, :]
y_train2 = y_train[pos_two]

# 02 (positive class is 2)
pos_three = np.where(np.isin(y_train, [0, 2]))[0]
X_train3 = X_train[pos_three, :]
y_train3 = y_train[pos_three]


# In[225]:


params = {
  'C':[1e-4, 1e-3, 1e-2, 0.1, 1, 2, 4, 8, 10, 100],
}

svc1 =  SVC(kernel='linear', random_state=0)
clf1 = GridSearchCV(svc1, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3)
clf1.fit(X_train1, y_train1)

svc2 =  SVC(kernel='linear', random_state=0)
clf2 = GridSearchCV(svc2, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3)
clf2.fit(X_train2, y_train2)

svc3 =  SVC(kernel='linear', random_state=0)
clf3 = GridSearchCV(svc3, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3)
clf3.fit(X_train3, y_train3)

print(clf1.best_params_, clf2.best_params_, clf3.best_params_)


# In[226]:


print('### Training time for one vs one ###')
start = timer()
svc1 = SVC(kernel='linear', C=1, random_state=0)
svc1.fit(X_train1, y_train1)

svc2 = SVC(kernel='linear',C=0.1, random_state=0)
svc2.fit(X_train2, y_train2)

svc3 = SVC(kernel='linear', C=0.1, random_state=0)
svc3.fit(X_train3, y_train3)

end = timer()
print(end-start)


# In[227]:


# Predictions on Test data
def predict(svm, test, label):
    d = np.dot(test, svm.coef_.flatten())+svm.intercept_
    org_dist = np.array(d)
    org_dist = 1/(1 + np.exp(-org_dist))
    d[d > 0] = np.unique(label)[1]
    d[d < 0] = np.unique(label)[0]
    return d, org_dist


# In[231]:


pred1, d1 = predict(svc1, X_test, y_train1)
pred2, d2 = predict(svc2, X_test, y_train2)
pred3, d3 = predict(svc3, X_test, y_train3)
test_pred_1 = pd.DataFrame(np.hstack((pred1[:, np.newaxis], pred2[:, np.newaxis], pred3[:, np.newaxis])))
proba = np.hstack((d1[:, np.newaxis], d2[:, np.newaxis], d3[:, np.newaxis]))

# pred1, d1 = predict(svc1, X_train, y_train1)
# pred2, d2 = predict(svc2, X_train, y_train2)
# pred3, d3 = predict(svc3, X_train, y_train3)
# train_pred_1 = pd.DataFrame(np.hstack((pred1[:, np.newaxis], pred2[:, np.newaxis], pred3[:, np.newaxis])))
# proba = np.hstack((d1[:, np.newaxis], d2[:, np.newaxis], d3[:, np.newaxis]))


# In[232]:


def max_rep(s):
    return s.value_counts().index[0]    


# In[233]:


test_pred_1['result'] = test_pred_1.apply(max_rep, axis=1)
train_pred_1['result'] = train_pred_1.apply(max_rep, axis=1)


# In[243]:


# print(classification_report(y_train3, svc3.predict(X_train3)))


# In[244]:


print('Test Accuracy: ', accuracy_score(y_test, test_pred_1['result'].values))
# print(classification_report(y_test, test_pred_1['result'].values))
# print('Training Accuracy: ', accuracy_score(y_train, train_pred_1['result'].values))
print(f1_score(y_test, test_pred_1['result'].values, average='macro'))


# In[293]:


# skplt.metrics.plot_roc(y_test, proba, plot_macro=False, plot_micro=False, figsize=(6,4))
# plt.title('1-vs-1 SVM')
# plt.show()


# ## Implement One Vs Rest Linear  SVM

# In[246]:


# 0 vs 1,2 => 0 vs 3
pos_one = np.where(np.isin(y_train, [1, 2]))[0]
y_train1 = np.array(y_train)
y_train1[pos_one] = 3

#1 vs 0,2 => 1 vs 4
pos_two = np.where(np.isin(y_train, [0, 2]))[0]
y_train2 = np.array(y_train)
y_train2[pos_two] = 4

#2 vs 0,1 => 2 vs 5
pos_three = np.where(np.isin(y_train, [0, 1]))[0]
y_train3 = np.array(y_train)
y_train3[pos_three] = 5


# In[247]:


svc1 =  SVC(kernel='linear', random_state=42)
clf1 = GridSearchCV(svc1, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3)
clf1.fit(X_train, y_train1)

svc2 =  SVC(kernel='linear', random_state=42)
clf2 = GridSearchCV(svc2, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3)
clf2.fit(X_train, y_train2)

svc3 =  SVC(kernel='linear', random_state=42)
clf3 = GridSearchCV(svc3, param_grid=params, scoring='accuracy', n_jobs=-1, cv=3)
clf3.fit(X_train, y_train3)

print(clf1.best_params_, clf2.best_params_, clf3.best_params_)


# In[248]:


print('### Training time for one vs rest ###')
start = timer()
svc1 = SVC(kernel='linear', C=2, random_state=42)
svc1.fit(X_train, y_train1)

svc2 = SVC(kernel='linear', C=1, random_state=42)
svc2.fit(X_train, y_train2)

svc3 = SVC(kernel='linear', C=8, random_state=42)
svc3.fit(X_train, y_train3)

end = timer()
print(end-start)


# In[249]:


pred1, d1 = predict(svc1, X_test, y_train1)
pred2, d2 = predict(svc2, X_test, y_train2)
pred3, d3 = predict(svc3, X_test, y_train3)
test_pred_2 = pd.DataFrame(np.hstack((pred1[:, np.newaxis], pred2[:, np.newaxis], pred3[:, np.newaxis])))
proba = pd.DataFrame(np.hstack((d1[:, np.newaxis], d2[:, np.newaxis], d3[:, np.newaxis])))

# pred1, d1 = predict(svc1, X_train, y_train1)
# pred2, d2 = predict(svc2, X_train, y_train2)
# pred3, d3 = predict(svc3, X_train, y_train3)
# train_pred_2 = pd.DataFrame(np.hstack((pred1[:, np.newaxis], pred2[:, np.newaxis], pred3[:, np.newaxis])))
# proba = pd.DataFrame(np.hstack((d1[:, np.newaxis], d2[:, np.newaxis], d3[:, np.newaxis])))


# In[250]:


def max_rep2(s):
    return s.idxmin()


# In[251]:


test_pred_2['result'] = proba.apply(max_rep2, axis=1)
# train_pred_1['result'] = train_pred_1.apply(max_rep, axis=1)


# In[252]:


print('Accuracy: ', accuracy_score(y_test, test_pred_2['result'].values))
# print('Accuracy: ', accuracy_score(y_train, train_pred_2['result'].values))
# print(classification_report(y_test, test_pred_2['result'].values))
print(f1_score(y_test, test_pred_2['result'].values, average='macro'))


# In[261]:


# print(classification_report(y_train1, svc1.predict(X_train)))


# In[294]:


# skplt.metrics.plot_roc(y_test, proba, plot_macro=False, plot_micro=False, figsize=(6,4))
# plt.title('1-vs-rest SVM')
# plt.show()


# ## Gaussian Naive bayes 

# In[283]:


clf = GaussianNB()
start = timer()
clf.fit(X_train, y_train)
end = timer()
print(end-start)


# In[284]:


print(accuracy_score(y_train, clf.predict(X_train)))


# In[285]:


print(classification_report(y_test, clf.predict(X_test)))


# In[287]:


skplt.metrics.plot_roc(y_test, clf.predict_proba(X_test), plot_macro=False, plot_micro=False, figsize=(6,4))
plt.title('Gaussian Naive Bayes ROC')
plt.show()
print(f1_score(y_test, clf.predict(X_test), average='macro'))


# ## Decision Trees

# In[276]:


params = {
    'max_depth':[3, 4, 5, 7, 8, 9, None],
    'min_samples_split':[2,3,4,5,6],
    'min_samples_leaf':[2,3,4,5,6]
    
}
start = timer()
# dt = DecisionTreeClassifier(max_depth = 3, random_state=0)
clf = GridSearchCV(dt, param_grid=params, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)
end = timer()
print(end - start)


# In[277]:


print(clf.best_estimator_)
# pd.DataFrame(clf.cv_results_)


# In[280]:


print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
# clf.predict_proba(X_test)


# In[215]:


# print(dt.score(X_train, y_train))


# In[216]:


# print(f1_score(y_test, dt.predict(X_test), average='macro'))


# In[281]:


skplt.metrics.plot_roc(y_test, clf.predict_proba(X_test), plot_macro=False, plot_micro=False, figsize=(6,4))
plt.title('Decision Tree ROC')
plt.show()
print(f1_score(y_test, clf.predict(X_test), average='macro'))


# In[ ]:




