#!/usr/bin/env python
# coding: utf-8
import np
# In[35]:


#import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

train_data['train_test'] = 1
test_data['train_test'] = 0
test_data['Survived'] = np.NaN
all_data = pd.concat([train_data,test_data])

# Display the first few rows of the training data for an overview
train_data.head()


# In[5]:


train_data.info()


# In[7]:


#look at numeric and categorical values seperately
df_num = train_data[['Age','SibSp','Parch','Fare']]
df_cat = train_data[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]


# In[8]:


# distribution for all numeric variables
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[9]:


print(df_num.corr())
sns.heatmap(df_num.corr())


# In[11]:


# compare survival rate across Age, SibSp, Parch, and Fare 
pd.pivot_table(train_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])


# In[14]:


for i in df_cat.columns:
    # sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    sns.barplot(x=df_cat[i].value_counts().index, y=df_cat[i].value_counts()).set_title(i)

    plt.show()


# In[15]:


# Comparing survival and each of these categorical variables 
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(train_data, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))


# In[16]:


# Feature engineering
# Ticket and cabin are messy, need to do feature engineering
df_cat.Cabin
train_data['cabin_multiple'] = train_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
# after looking at this, we may want to look at cabin by letter or by number. Let's create some categories for this 
# multiple letters 
train_data['cabin_multiple'].value_counts()


# In[18]:


pd.pivot_table(train_data, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket' ,aggfunc ='count')


# In[19]:


#creates categories based on the cabin letter (n stands for null)
#in this case we will treat null values like it's own category

train_data['cabin_adv'] = train_data.Cabin.apply(lambda x: str(x)[0])


# In[20]:


#comparing surivial rate by cabin
print(train_data.cabin_adv.value_counts())
pd.pivot_table(train_data,index='Survived',columns='cabin_adv', values = 'Name', aggfunc='count')


# In[22]:


#understand ticket values better 
#numeric vs non numeric 
train_data['numeric_ticket'] = train_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
train_data['ticket_letters'] = train_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)


# In[24]:


train_data['numeric_ticket'].value_counts()


# In[25]:


#lets us view all rows in dataframe through scrolling. This is for convenience 
pd.set_option("display.max_rows", None)
train_data['ticket_letters'].value_counts()


# In[27]:


#difference in numeric vs non-numeric tickets in survival rate 
pd.pivot_table(train_data,index='Survived',columns='numeric_ticket', values = 'Ticket', aggfunc='count')


# In[28]:


#survival rate across different tyicket types 
pd.pivot_table(train_data,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')


# In[31]:


#feature engineering on person's title 
train_data.Name.head(50)
train_data['name_title'] = train_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
#mr., ms., master. etc
train_data['name_title'].value_counts()


# In[38]:


# Data processing
#create all categorical variables that we did above for both training and test sets 
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data 
#all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Age = all_data.Age.fillna(train_data.Age.median())
#all_data.Fare = all_data.Fare.fillna(training.Fare.mean())
all_data.Fare = all_data.Fare.fillna(train_data.Fare.median())

#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 
all_data.dropna(subset=['Embarked'],inplace = True)

#tried log norm of sibsp (not used)
all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()

# log norm of fare (used)
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()

# converted fare to category for pd.get_dummies()
all_data.Pclass = all_data.Pclass.astype(str)

#created dummy variables from categories (also can use OneHotEncoder)
all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])

#Split to train test again
X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)


y_train = all_data[all_data.train_test==1].Survived
y_train.shape


# In[39]:


# Scale data 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived


# In[41]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
#
# # Define the model
# log_model = LogisticRegression()
#
# # Define the parameters grid to search
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
#     'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Algorithm to use for optimization
#     'max_iter': [1000]  # Iterations to ensure convergence
# }
#
# # Initialize GridSearchCV with the logistic regression model, parameter grid, and desired number of folds for cross-validation
# grid_search = GridSearchCV(log_model, param_grid, cv=5, verbose=1, n_jobs=-1)
#
# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
#
# # Print out the best parameters and the best score achieved
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validated score: {:.2f}".format(grid_search.best_score_))
#
# # You can also get the best estimator directly
# best_model = grid_search.best_estimator_
#
#
# # In[46]:
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import learning_curve, cross_val_score
#
# # Assuming X_train and y_train are already defined and preprocessed
# log_model = LogisticRegression(C=100, max_iter=1000, solver='newton-cg')
#
# # Fit the model
# log_model.fit(X_train, y_train)
#
# # Perform cross-validation
# cv_scores = cross_val_score(log_model, X_train, y_train, cv=5)
# cv_errors = 1 - cv_scores
#
# # Print cross-validation results
# print(f"Cross-validation accuracy scores: {cv_scores}")
# print(f"Mean CV accuracy: {np.mean(cv_scores):.2f}")
# print(f"Cross-validation error rates: {cv_errors}")
# print(f"Mean CV error rate: {np.mean(cv_errors):.2f}")
#
# # Generate learning curves
# train_sizes, train_scores, test_scores = learning_curve(
#     log_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
# )
#
# # Calculate mean and standard deviation for training and test scores
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# train_errors_mean = 1 - train_scores_mean
# train_errors_std = train_scores_std
#
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# test_errors_mean = 1 - test_scores_mean
# test_errors_std = test_scores_std
#
# # Plot learning curve for accuracy
# plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.title("Learning Curve (Accuracy)")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.grid()
#
# # Plot learning curve for error
# plt.subplot(1, 2, 2)
# plt.fill_between(train_sizes, train_errors_mean - train_errors_std, train_errors_mean + train_errors_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_errors_mean - test_errors_std, test_errors_mean + test_errors_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Training error")
# plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Cross-validation error")
# plt.title("Learning Curve (Error)")
# plt.xlabel("Training examples")
# plt.ylabel("Error Rate")
# plt.legend(loc="best")
# plt.grid()
#
# plt.tight_layout()
# plt.show()
#
#
# # In[48]:
#
#
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV
#
# # Assuming X_train and y_train are already defined and preprocessed
# dt_model = DecisionTreeClassifier()
#
# # Define the parameter grid to search
# param_grid = {
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'criterion': ['gini', 'entropy']
# }
#
# # Initialize GridSearchCV with the decision tree model, parameter grid, and number of folds for cross-validation
# grid_search = GridSearchCV(dt_model, param_grid, cv=5, verbose=1, n_jobs=-1)
#
# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
#
# # Print out the best parameters and the best score achieved
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validated score: {:.2f}".format(grid_search.best_score_))
#
# # You can also get the best estimator directly
# best_dt_model = grid_search.best_estimator_
#
#
# # In[49]:
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import learning_curve, cross_val_score
#
# # Assuming X_train and y_train are already defined and preprocessed
# dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2, min_samples_split=10)
#
# # Fit the model
# dt_model.fit(X_train, y_train)
#
# # Perform cross-validation
# cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5)
# cv_errors = 1 - cv_scores
#
# # Print cross-validation results
# print(f"Cross-validation accuracy scores: {cv_scores}")
# print(f"Mean CV accuracy: {np.mean(cv_scores):.2f}")
# print(f"Cross-validation error rates: {cv_errors}")
# print(f"Mean CV error rate: {np.mean(cv_errors):.2f}")
#
# # Generate learning curves
# train_sizes, train_scores, test_scores = learning_curve(
#     dt_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
# )
#
# # Calculate mean and standard deviation for training and test scores
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# train_errors_mean = 1 - train_scores_mean
# train_errors_std = train_scores_std
#
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# test_errors_mean = 1 - test_scores_mean
# test_errors_std = test_scores_std
#
# # Plot learning curve for accuracy
# plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.title("Learning Curve (Accuracy) for Decision Tree")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.grid()
#
# # Plot learning curve for error
# plt.subplot(1, 2, 2)
# plt.fill_between(train_sizes, train_errors_mean - train_errors_std, train_errors_mean + train_errors_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_errors_mean - test_errors_std, test_errors_mean + test_errors_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Training error")
# plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Cross-validation error")
# plt.title("Learning Curve (Error) for Decision Tree")
# plt.xlabel("Training examples")
# plt.ylabel("Error Rate")
# plt.legend(loc="best")
# plt.grid()
#
# plt.tight_layout()
# plt.show()
#
#
# # In[51]:
#
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
#
# # Define the KNN model
# knn_model = KNeighborsClassifier()
#
# # Define the parameter grid
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
#
# # Initialize GridSearchCV
# grid_search = GridSearchCV(knn_model, param_grid, cv=5, verbose=1, n_jobs=-1)
#
# # Assuming X_train and y_train are your data
# # Fit the grid search
# grid_search.fit(X_train, y_train)
#
# # Print the best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validated score: {:.2f}".format(grid_search.best_score_))
#
# # Best model
# best_knn_model = grid_search.best_estimator_
#
#
# # In[52]:
#
#
# cv_scores = cross_val_score(best_knn_model, X_train, y_train, cv=5)
# cv_errors = 1 - cv_scores
#
# # Print cross-validation results
# print(f"Cross-validation accuracy scores: {cv_scores}")
# print(f"Mean CV accuracy: {np.mean(cv_scores):.2f}")
# print(f"Cross-validation error rates: {cv_errors}")
# print(f"Mean CV error rate: {np.mean(cv_errors):.2f}")
#
#
# # In[55]:
#
#
# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Generate learning curves
# train_sizes, train_scores, test_scores = learning_curve(
#     best_knn_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
# )
#
# # Calculate mean and standard deviation for training and test scores
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# train_errors_mean = 1 - train_scores_mean
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# test_errors_mean = 1 - test_scores_mean
#
# # Plot learning curve for accuracy
# plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.title("Learning Curve (Accuracy) for KNN")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.grid()
#
# # Plot learning curve for error
# plt.subplot(1, 2, 2)
# plt.fill_between(train_sizes, train_errors_mean, train_errors_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_errors_mean, test_errors_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Training error")
# plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Cross-validation error")
# plt.title("Learning Curve (Error) for KNN")
# plt.xlabel("Training examples")
# plt.ylabel("Error Rate")
# plt.legend(loc="best")
# plt.grid()
#
# plt.tight_layout()
# plt.show()
#
#
# # In[56]:
#
#
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import GridSearchCV
#
# # Define the Gradient Boosting model
# gb_model = GradientBoostingClassifier()
#
# # Define the parameter grid to search
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 4, 5],
#     'min_samples_split': [2, 3, 4],
#     'min_samples_leaf': [1, 2, 3]
# }
#
# # Initialize GridSearchCV
# grid_search = GridSearchCV(gb_model, param_grid, cv=5, verbose=1, n_jobs=-1)
#
# # Assuming X_train and y_train are already defined and preprocessed
# grid_search.fit(X_train, y_train)
#
# # Print the best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validated score: {:.2f}".format(grid_search.best_score_))
#
# # Best model
# best_gb_model = grid_search.best_estimator_
#
#
# # In[57]:
#
#
# #Training the Gradient Boosting Model with the Best Parameters
# best_gb_model.fit(X_train, y_train)
#
#
# # In[58]:
#
#
# #cross-validation
# from sklearn.model_selection import cross_val_score
# import numpy as np
#
# cv_scores = cross_val_score(best_gb_model, X_train, y_train, cv=5)
# print(f"Cross-validation accuracy scores: {cv_scores}")
# print(f"Mean CV accuracy: {np.mean(cv_scores):.2f}")
#
#
# # In[60]:
#
#
# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt
# import numpy as np
#
# train_sizes, train_scores, test_scores = learning_curve(
#     best_gb_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
# )
#
# # Calculate mean and standard deviation for training and test scores
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# # Calculate error rates
# train_errors_mean = 1 - train_scores_mean
# test_errors_mean = 1 - test_scores_mean
#
# # Plot learning curve for accuracy
# plt.figure(figsize=(14, 6))
#
# plt.subplot(1, 2, 1)
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.title("Learning Curve (Accuracy) for Gradient Boosting")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.grid()
#
# # Plot learning curve for error
# plt.subplot(1, 2, 2)
# plt.fill_between(train_sizes, train_errors_mean, train_errors_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_errors_mean, test_errors_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Training error")
# plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Cross-validation error")
# plt.title("Learning Curve (Error) for Gradient Boosting")
# plt.xlabel("Training examples")
# plt.ylabel("Error Rate")
# plt.legend(loc="best")
# plt.grid()
#
# plt.tight_layout()
# plt.show()
#
#
# # In[ ]:
#
#
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
#
# # Define the SVM model
# svm_model = SVC()
#
# # Define the parameter grid to search
# param_grid = {
#     'C': [0.1, 1, 10, 100],  # Regularization parameter
#     'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
#     'kernel': ['rbf', 'poly', 'sigmoid']
# }
#
# # Initialize GridSearchCV
# grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=1, n_jobs=-1)
#
# # Assuming X_train and y_train are already defined and preprocessed
# grid_search.fit(X_train, y_train)
#
# # Print the best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validated score: {:.2f}".format(grid_search.best_score_))
#
# # Best model
# best_svm_model = grid_search.best_estimator_
#
#
# # In[ ]:
#
#
# #Training the SVM Model with the Best Parameters
# best_svm_model.fit(X_train, y_train)
#
#
# # In[ ]:
#
#
# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt
# import numpy as np
#
# train_sizes, train_scores, test_scores = learning_curve(
#     best_svm_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
# )
#
# # Calculate mean and standard deviation for training and test scores
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# # Calculate error rates
# train_errors_mean = 1 - train_scores_mean
# test_errors_mean = 1 - test_scores_mean
#
# # Plot learning curve for accuracy
# plt.figure(figsize=(14, 6))
#
# plt.subplot(1, 2, 1)
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.title("Learning Curve (Accuracy) for SVM")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.grid()
#
# # Plot learning curve for error
# plt.subplot(1, 2, 2)
# plt.fill_between(train_sizes, train_errors_mean, train_errors_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_errors_mean, test_errors_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_errors_mean, 'o-', color="r", label="Training error")
# plt.plot(train_sizes, test_errors_mean, 'o-', color="g", label="Cross-validation error")
# plt.title("Learning Curve (Error) for SVM")
# plt.xlabel("Training examples")
# plt.ylabel("Error Rate")
# plt.legend(loc="best")
# plt.grid()
#
# plt.tight_layout()
# plt.show()

# 创建包含所有模型及其参数网格的字典
models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'max_iter': [1000]
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 50, 100, None]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 1],
            'max_depth': [3, 5, 7]
        }
    },
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly']
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
}

# 假设 X_train, y_train 是你的训练数据
# X_train, y_train = ...

# 循环遍历模型字典
for model_name, model_info in models.items():
    print(f"Training {model_name}...")

    model = model_info['model']
    param_grid = model_info['params']

    # 使用 GridSearchCV 来寻找最佳参数
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best score for {model_name}: {grid_search.best_score_}")


# # Preparing the data for PCA and SVC
# # Selecting features and target variable
# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
# X = train_df[features]
# y = train_df['Survived']
#
# # Handling categorical data and missing values
# X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
# X['Age'].fillna(X['Age'].mean(), inplace=True)
#
# # Standardizing the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Applying PCA
# pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
# X_pca = pca.fit_transform(X_scaled)
#
# # Training SVC with linear kernel
# svc = SVC(kernel='linear')
# svc.fit(X_pca, y)
#
# # Visualizing the results
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.5)
#
# # Plotting the decision boundary
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
# # Creating grid to evaluate model
# xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
#                      np.linspace(ylim[0], ylim[1], 50))
# Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# # Plot decision boundary and margins
# ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
#
# # Plot support vectors
# ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none', edgecolors='k')
#
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('SVC with Linear Kernel on PCA-transformed Data')
# plt.show()
