# Disease-Prediction-ML

Code :
`````
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("/content/Training.csv")
test = pd.read_csv("/content/Testing.csv")

train.head(2)
test.head(2)
train= train.drop(["Unnamed: 133"],axis=1)
train.prognosis.value_counts()
Y = train[["prognosis"]]
X = train.drop(["prognosis"],axis=1)
P = test.drop(["prognosis"],axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42

#Model 1 - Decision Tree
dtc= DecisionTreeClassifier(random_state=42)
model_dtc = dtc.fit(xtrain,ytrain)
tr_pred_dtc = model_dtc.predict(xtrain)
ts_pred_dtc = model_dtc.predict(xtest)

print("training accuracy is:",accuracy_score(ytrain,tr_pred_dtc))
print("testing accuracy is:",accuracy_score(ytest,ts_pred_dtc))

# Model 2 - Random Forest
rfc= RandomForestClassifier(random_state=42)
model_rfc = rfc.fit(xtrain,ytrain)
tr_pred_rfc = model_rfc.predict(xtrain)
ts_pred_rfc = model_rfc.predict(xtest)

print("training accuracy is:",accuracy_score(ytrain,tr_pred_rfc))
print("testing accuracy is:",accuracy_score(ytest,ts_pred_rfc))

test.join(pd.DataFrame(model_dtc.predict(P),columns=["predicted"]))[["prognosis","predicted"]]
`````

