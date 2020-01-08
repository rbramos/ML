# This data was extracted from the census bureau database found at
# http://www.census.gov/ftp/pub/DES/www/welcome.html
# Donor: Ronny Kohavi and Barry Becker,
#        Data Mining and Visualization
#        Silicon Graphics.
#        e-mail: ronnyk@sgi.com for questions.
# Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
# 48842 instances, mix of continuous and discrete    (train=32561, test=16281)
# 45222 if instances with unknown values are removed (train=30162, test=15060)
# Duplicate or conflicting instances : 6
# Class probabilities for adult.all file
# Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
# Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
#
# Extraction was done by Barry Becker from the 1994 Census database.  A set of
#   reasonably clean records was extracted using the following conditions:
#   ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
#
# Prediction task is to determine whether a person makes over 50K
# a year.
#
# libraries import

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score

# import data
db = pd.read_csv('./adult.data')

descriptive = db.iloc[ : , :14].values
target = db.iloc[ : , 14].values

# # defining the objects to process the data
labelEncoder = LabelEncoder()

# # split the columns to create the encoders
#descriptive[:, 1] = labelEncoder.fit_transform(descriptive[:, 1])
#descriptive[:, 3] = labelEncoder.fit_transform(descriptive[:, 3])
#descriptive[:, 5] = labelEncoder.fit_transform(descriptive[:, 5])
#descriptive[:, 6] = labelEncoder.fit_transform(descriptive[:, 6])
#descriptive[:, 7] = labelEncoder.fit_transform(descriptive[:, 7])
#descriptive[:, 8] = labelEncoder.fit_transform(descriptive[:, 8])
#descriptive[:, 9] = labelEncoder.fit_transform(descriptive[:, 9])
#descriptive[:,13] = labelEncoder.fit_transform(descriptive[:,13])

# use the One Hot encoder
oneHotEncoder = ColumnTransformer([("one_hot_encoder",OneHotEncoder(categories='auto'), [1, 3, 5, 6, 7, 8, 9, 13] )], remainder="passthrough")

descriptive = oneHotEncoder.fit_transform(descriptive)

# # create the standard scaler
standardScaler = StandardScaler(with_mean=False)
descriptive = standardScaler.fit_transform(descriptive).toarray()


# # split the descriptive and target to train and test
descriptiveTraining, descriptiveTest, targetTraining, targetTest = train_test_split(descriptive, target, test_size = 0.25, random_state = 0)
classifier = GaussianNB()
classifier.fit(descriptiveTraining, targetTraining)

# # predict the values
prediction = classifier.predict(descriptiveTest)

# # show the accurary of the predictions
matrix = confusion_matrix(targetTest, prediction)
accuracy = accuracy_score(targetTest, prediction)

print(matrix)
print(accuracy)
