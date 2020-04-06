"This file will be used for model training and serialization"

import pandas as pd
pd.set_option('display.width', 10000)
pd.options.display.max_columns = 100
from xgboost import XGBRegressor as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# Pre-trained model and scaler
FILENAME = 'model'
OUTFILE = open(FILENAME, 'wb')
SCALE = 'scaler'
SCALER = open(SCALE, 'wb')


#  Init dataframe
df = pd.read_csv(
    '/Users/nickalonso/PycharmProjects/techtalk-test/kml-bbox-tutorial/data/airbnb_historical_listings.csv')



# Feature set and target variable
x = df[['accommodates','bedrooms','bathrooms','cleaning_fee','distance','size']]
y = df['price']



# Create train/test set with scaling applied
sc = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x_train = sc.fit_transform(x_train)
pickle.dump(sc, SCALER)
SCALER.close()
x_test = sc.transform(x_test)



# Xgboost parameters hard coded after grid-search cross validation
booster = xgb(n_estimators=200,random_state=4,gamma=0.2,max_depth=6,learning_rate=0.1,colsample_bytree=0.7)



# Fit model and serialize
booster.fit(x_train,y_train)
pickle.dump(booster,OUTFILE)
OUTFILE.close()



# Validate model
y_pred_test=booster.predict(x_test)
for i in y_pred_test:
    print("$",round(i,2),"/ night")

