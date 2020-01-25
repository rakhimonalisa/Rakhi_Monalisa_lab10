# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
path = "C:/Users/300997447/Desktop/New folder"
filename = 'iris.csv'
fullpath = os.path.join(path,filename)
data_mayy_i = pd.read_csv(fullpath,sep=',')
print(data_mayy_i.columns.values)
print(data_mayy_i.shape)
print(data_mayy_i.describe())
print(data_mayy_i.dtypes) 
print(data_mayy_i.head(5))
print(data_mayy_i['Species'].unique())

################################3

colnames=data_mayy_i.columns.values.tolist()
predictors=colnames[:4]
target=colnames[4]
import numpy as np
data_mayy_i['is_train'] = np.random.uniform(0, 1, len(data_mayy_i)) <= .75
print(data_mayy_i.head(5))
# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_mayy_i[data_mayy_i['is_train']==True], data_mayy_i[data_mayy_i['is_train']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

########################4

from sklearn.tree import DecisionTreeClassifier
dt_mayy = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_mayy.fit(train[predictors], train[target])

########################5

preds=dt_mayy.predict(test[predictors])
pd.crosstab(test['Species'],preds,rownames=['Actual'],colnames=['Predictions'])

##########################6
for depth in range(1,11):
    
    from sklearn.tree import export_graphviz
    with open('C:/Users/300997447/Desktop/New folder/dtree3.dot', 'w') as dotfile:
        export_graphviz(dt_mayy, out_file = dotfile, feature_names = predictors)
    dotfile.close()
    
    ##########################7
    
    X=data_mayy_i[predictors]
    Y=data_mayy_i[target]
    #split the data sklearn module
    from sklearn.model_selection import train_test_split
    trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)
    
    #############################8
    for depth in range(1,11):
     dt1_mayy = DecisionTreeClassifier(criterion='entropy',max_depth=depth, min_samples_split=20, random_state=99)
    #dt1_mayy = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20, random_state=99)
    dt1_mayy.fit(trainX,trainY)
    # 10 fold cross validation using sklearn and all the data i.e validate the data 
    from sklearn.model_selection import KFold
    #help(KFold)
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    from sklearn.model_selection import cross_val_score
    score = np.mean(cross_val_score(dt1_mayy, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
    print("max_depth = ",depth,"score=",score)
    # print (score)
    
    
        
    
    ######################9
    
    ### Test the model using the testing data
    testY_predict = dt1_mayy.predict(testX)
    testY_predict.dtype
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics 
    labels = Y.unique()
    print(labels)
    print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
    #Let us print the confusion matrix
    from sklearn.metrics import confusion_matrix
    print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))
    
    #########################10
    
    import seaborn as sns
    import matplotlib.pyplot as plt     
    cm = confusion_matrix(testY, testY_predict, labels)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']); ax.yaxis.set_ticklabels(['setosa', 'versicolor', 'virginica']);
    plt.show()
    print ("feature importance")
    #print(data_mayy_i.columns.values)
    print(dt1_mayy.feature_importances_)


