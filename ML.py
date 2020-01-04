# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.compose import ColumnTransformer # FutureWarning: The handling of integer data will change in version 0.22

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np

class PreProcessing:
    
    nColunas = 0
    categoricalList = []
    
    def readFile(self,file):
        dataset = pd.read_csv(file)
        return dataset
    
#    
    def cleanDataset(self, dataset):
        imp = SimpleImputer( missing_values ="?", strategy="most_frequent")
        dataset=imp.fit_transform(dataset).toarray()
        #dataset=np.array(dataset)
        return dataset
        
        
    def processingDescriptiveTarget(self,dataset):
        #dataset.describe()
        num_cols = len(dataset.columns) - 1
        self.nColunas = num_cols
        descriptive = dataset.iloc[:,0:num_cols].values
        target = dataset.iloc[:,num_cols].values
        
        result = { 'descriptive': descriptive, 'target' : target }
        return  result
    
    def getData(self,file):
        dataset = self.readFile(file)
        dataset = self.cleanDataset(dataset)
        DT = self.processingDescriptiveTarget(dataset)
        
        return DT
    
    #Label Encoder and One Hot Encoder. These two encoders are parts of the 
    #SciKit Learn library in Python, and they are used to convert categorical data, 
    #or text data, into numbers, which our predictive models can better understand
    #LabelEncoder is a utility class to help normalize labels such that they 
    #contain only values between 0 and n_classes-1
    #https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
    def labelEncoder(self,descriptive):
        le = LabelEncoder()
        
        for i in range(self.nColunas - 1):
            descriptive[:,i] = le.fit_transform(descriptive[:,0])
            self.categoricalList.append(i)
        
        return descriptive
    
    def oneHotEncoder(self,descriptive):
        #he = OneHotEncoder(categorical_features = self.categoricalList)
        he = ColumnTransformer([("one_hot_encoder",OneHotEncoder(categories='auto'), self.categoricalList)], remainder="passthrough") #kill warning
        descriptive = he.fit_transform(descriptive).toarray()
        
        return descriptive
    
    #standardization
    def standarScaler(self,descriptive):
        ss = StandardScaler()
        descriptive = ss.fit_transform(descriptive)
        
        return descriptive
    
class Processing:
    def splitDataset(self,descriptive,target,testSize,randomState):
        descriptiveTraining, descriptiveTest, targetTraining, targetTest = train_test_split(descriptive, target, test_size = testSize, random_state=randomState)
        result = {"descriptiveTraining":descriptiveTraining, "descriptiveTest" : descriptiveTest,  "targetTraining" : targetTraining, "targetTest" : targetTest}
        return result
   
    def naiveBayes(self):
        classifier = GaussianNB()
        return classifier
    
    def decisionTree(self):
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        return classifier
    
    def randomForest(self):
        classifier = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0)
        return classifier

    
    def kNN(self, numberNeighbors):
        numberNeighbors = int(numberNeighbors)
        classifier = KNeighborsClassifier(n_neighbors=numberNeighbors, metric='minkowski', p=2)
        return classifier
        
    def getPrediction(self, classifier, descriptiveTraining, descriptiveTest ,targetTraining):
        classifier.fit(descriptiveTraining, targetTraining)
        prediction = classifier.predict(descriptiveTest)
        
        return prediction
    
    def getResults(self, targetTest, prediction):
        accuracy = accuracy_score(targetTest, prediction)
        matrix = confusion_matrix(targetTest, prediction)
        
        result = {"accuracy":accuracy, "matrix":matrix}
        
        return result
    
    

        
    
   
    
        