# Reference : https://www.kaggle.com/chapagain/titanic-solution-a-beginner-s-guide

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('mem_train.csv')
test = pd.read_csv('mem_test.csv')

# ZIP_CD
# 0 to '-'
# 1 otherwise
train.loc[train.ZIP_CD == '-', 'ZIP_CD'] = 0
test.loc[test.ZIP_CD == '-', 'ZIP_CD'] = 0
train.loc[train.ZIP_CD != '-', 'ZIP_CD'] = 1
test.loc[test.ZIP_CD != '-', 'ZIP_CD'] = 1
train.ZIP_CD = train.ZIP_CD.astype(int)
test.ZIP_CD = test.ZIP_CD.astype(int)

# Drop irrelevant(or naughty) columns
train = train.drop(['RGST_DT', 'LAST_VST_DT'], axis=1)
test = test.drop(['RGST_DT', 'LAST_VST_DT'], axis=1)

# GENDER
# F = 0, M = 1
train.loc[train.GENDER == 'F', 'GENDER'] = 0
train.loc[train.GENDER == 'M', 'GENDER'] = 1
train.GENDER = train.GENDER.astype(int)

# BIRTH_SL
# S = 0, L = 1
train.loc[train.BIRTH_SL == 'S', 'BIRTH_SL'] = 0
train.loc[train.BIRTH_SL == 'L', 'BIRTH_SL'] = 1
test.loc[test.BIRTH_SL == 'S', 'BIRTH_SL'] = 0
test.loc[test.BIRTH_SL == 'L', 'BIRTH_SL'] = 1
train.BIRTH_SL = train.BIRTH_SL.astype(int)
test.BIRTH_SL = test.BIRTH_SL.astype(int)

# SMS
# Y = 0, N = 1
train.loc[train.SMS == 'Y', 'SMS'] = 0
train.loc[train.SMS == 'N', 'SMS'] = 1
test.loc[test.SMS == 'Y', 'SMS'] = 0
test.loc[test.SMS == 'N', 'SMS'] = 1
train.SMS = train.SMS.astype(int)
test.SMS = test.SMS.astype(int)

# Processing BIRTH_DT

# 0. fillna with 9999-99-99 for my convenience.
train["BIRTH_DT"].fillna('9999-99-99', inplace = True)
test["BIRTH_DT"].fillna('9999-99-99', inplace = True)

# 1. YYYY-MM-DD to YYYY
for ind in train.index:
    if len(str(train['BIRTH_DT'].iloc[ind])) < 6:
        train['BIRTH_DT'].iloc[ind] = 9999
    else:
        train['BIRTH_DT'][ind] = train['BIRTH_DT'][ind][:4]
train.BIRTH_DT = train.BIRTH_DT.astype(int)

for ind in test.index:
    if len(str(test['BIRTH_DT'][ind])) < 6:
        test['BIRTH_DT'][ind] = 9999
    else:
        test['BIRTH_DT'][ind] = test['BIRTH_DT'][ind][:4]
test.BIRTH_DT = test.BIRTH_DT.astype(int)

#print(train.BIRTH_DT.value_counts())

# 1-5. Deal with strange years based on my own heuristic.
train.loc[train['BIRTH_DT'] > 2020, 'BIRTH_DT'] = 9999
train.loc[train['BIRTH_DT'] < 1900, 'BIRTH_DT'] = 9999
test.loc[test['BIRTH_DT'] > 2020, 'BIRTH_DT'] = 9999
test.loc[test['BIRTH_DT'] < 1900, 'BIRTH_DT'] = 9999

# 2. Replacing 9999 with a more plausible number :)
# We see the missing cases at BIRTH_DT as MCAR(see train.corr() without rows with 9999).
# So, we will replace 9999 with a proper year based on the distribution of the years.
train_years = train.loc[train['BIRTH_DT'] != 9999].BIRTH_DT
train_years = train_years.value_counts(normalize=True)
#print(train_years)
train_len_9999 = len(train.loc[train.BIRTH_DT == 9999, 'BIRTH_DT'])
train.loc[train.BIRTH_DT == 9999, 'BIRTH_DT'] = np.random.choice(train_years.index, size=train_len_9999, p=train_years.values)
#print(train.BIRTH_DT.value_counts())

test_years = test.loc[test['BIRTH_DT'] != 9999].BIRTH_DT
test_years = test_years.value_counts(normalize=True)
test_len_9999 = len(test.loc[test.BIRTH_DT == 9999, 'BIRTH_DT'])
test.loc[test.BIRTH_DT == 9999, 'BIRTH_DT'] = np.random.choice(test_years.index, size=test_len_9999, p=test_years.values)

#print(train.info())
#print(test.info())






# Define training and testing set
X_train = train.drop(['MEM_ID', 'GENDER'], axis=1)
y_train = train['GENDER']
X_test = test.drop('MEM_ID', axis=1)
#print(X_train.shape, y_train.shape,X_test.shape)

# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

# Train-Validation Data Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.05, random_state=777)

# ROC AUC Score Module
from sklearn.metrics import roc_auc_score

# 1. Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
acc_log_reg_val = round( clf.score(X_val, y_val) * 100, 2)
print ('Logistic Regression: ' + str(acc_log_reg) + ' percent')
print ('Logistic Regression_val: ' + str(acc_log_reg_val) + ' percent')
print ('ROC AUC Score of Regression(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 2. Support Vector Machine(SVM)
clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
acc_svc_val = round(clf.score(X_val, y_val) * 100, 2)
print ('SVM: ' + str(acc_svc) + ' percent')
print ('SVM_val: ' + str(acc_svc_val) + ' percent')
print ('ROC AUC Score of SVM(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 3. Linear SVM
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
acc_linear_svc_val = round(clf.score(X_val, y_val) * 100, 2)
print ('Linear SVC: ' + str(acc_linear_svc) + ' percent')
print ('Linear SVC_val: ' + str(acc_linear_svc_val) + ' percent')
print ('ROC AUC Score of Linear SVC(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 4. KNN
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
acc_knn_val = round(clf.score(X_val, y_val) * 100, 2)
print ('KNN: ' + str(acc_knn) + ' percent')
print ('KNN_val: ' + str(acc_knn_val) + ' percent')
print ('ROC AUC Score of KNN(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 5. Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
acc_decision_tree_val = round(clf.score(X_val, y_val) * 100, 2)
print ('Decision Tree: ' + str(acc_decision_tree) + ' percent')
print ('Decision Tree_val: ' + str(acc_decision_tree_val) + ' percent')
print ('ROC AUC Score of Decision Tree(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 6. Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
acc_random_forest_val = round(clf.score(X_val, y_val) * 100, 2)
print ('Random Forest: ' + str(acc_random_forest) + ' percent')
print ('Random Forest_val: ' + str(acc_random_forest_val) + ' percent')
print ('ROC AUC Score of Random Forest(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 7. Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
acc_gnb_val = round(clf.score(X_val, y_val) * 100, 2)
print ('Naive Bayes: ' + str(acc_gnb) + 'percent')
print ('Naive Bayes_val: ' + str(acc_gnb_val) + 'percent')
print ('ROC AUC Score of Naive Bayes(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 8. Perceptron
clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)
acc_perceptron_val = round(clf.score(X_val, y_val) * 100, 2)
print ('Perceptron: ' + str(acc_perceptron) + ' percent')
print ('Perceptron_val: ' + str(acc_perceptron_val) + ' percent')
print ('ROC AUC Score of Perceptron(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

# 9. Stochastic Gradient Descent(SGD)
clf = SGDClassifier(max_iter=20, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
acc_sgd_val = round(clf.score(X_val, y_val) * 100, 2)
print ('SGD: ' + str(acc_sgd) + ' percent')
print ('SGD_val: ' + str(acc_sgd_val) + ' percent')
print ('ROC AUC Score of SGD(Val): ', format(roc_auc_score(y_val, clf.predict(X_val)), ".2f"))

models = pd.DataFrame(
        {'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
            'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
            'Perceptron', 'Stochastic Gradient Decent'],
        'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
            acc_knn, acc_decision_tree, acc_random_forest, acc_gnb, 
            acc_perceptron, acc_sgd]}
        )
print(models.sort_values(by='Score', ascending=False))

models_val = pd.DataFrame(
        {'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
            'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
            'Perceptron', 'Stochastic Gradient Decent'],
        'Val_Score': [acc_log_reg_val, acc_svc_val, acc_linear_svc_val, 
            acc_knn_val, acc_decision_tree_val, acc_random_forest_val, acc_gnb_val, 
            acc_perceptron_val, acc_sgd_val]}
        )
print(models_val.sort_values(by='Val_Score', ascending=False))

# My Model Choice
y_pred_final = y_pred_random_forest

submission = pd.DataFrame(
        {"MEM_Id": test["MEM_ID"],
        "GENDER": y_pred_final}
        )

submission.to_csv('submission.csv', index=False)


