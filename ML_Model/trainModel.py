# These libraries are used by the machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import joblib

# This is the data file, you can download this file from the github link provided.
wine=pd.read_csv("D:\Wine_ML_Web_Project\ML_Model\data.csv")
print("Dataset Loaded...")
# Performing train test split with stratified shuffle split.
train_set, test_set  = train_test_split(wine, test_size=0.2, random_state=42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(wine, wine['Alchohol']):
	strat_train_set = wine.loc[train_index]
	strat_test_set = wine.loc[test_index]
wine = strat_train_set.copy()
wine = strat_train_set.drop("Wine", axis=1)
wine_labels = strat_train_set["Wine"].copy()

# Random Forest Regressor is used for prediction.
model = RandomForestRegressor()
model.fit(wine, wine_labels)
print("Model Training Ends...")
test_features=strat_test_set.drop("Wine", axis=1)
test_labels=strat_test_set["Wine"].copy()
y_labels=model.predict(test_features)
x=list(y_labels)
y=list(test_labels)
accuracy=[]
for i in range(len(test_labels)):
	if x[i]>y[i]:
		accuracy.append((y[i]/x[i])*100)
	else:
		accuracy.append((x[i]/y[i])*100)
joblib.dump(model, "rf_model2.joblib")
print("Model Saved...")
acc=sum(accuracy)/len(x)
print("Final Accuracy of the Model: ", acc)
