import pandas as pd
pd.options.display.max_columns = 100
pd.set_option('display.width', 10000)
from xgboost import XGBRegressor as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def gradient_boosting(df):
    '''Xgboost model using sub set of features that have already been engineered to work,
     applies standard scaling and trains model, serializes model to disk with pickle and outputs metrics'''
    FILENAME='model'
    OUTFILE=open(FILENAME, 'wb')
    SCALE='scaler'
    SCALER=open(SCALE, 'wb')

    # Create df for model training
    y = df[['price']]
    x = df[['accommodates','bedrooms','bathrooms','cleaning_fee','distance','size']]

    # Typically at this stage we would conduct some form of feature exploration, selection
    # and feature engineering. For the sake of time during the recording we have already
    # performed minor feature analysis and selection. We have also already conducted
    # hyper-parameter tuning using grid search cross-validation and will be hard coding
    # those params for our xgboost model. We could extend this project and improve the
    # models accuracy by performing further feature engineering using various NLP techniques
    # but will stick to using some basic int data types as features to predict the target
    # variable price. With our pre-defined feature set we jump into model training.

    # Create training/test set for training
    sc = StandardScaler()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    x_train=sc.fit_transform(x_train)
    pickle.dump(sc, SCALER)
    SCALER.close()
    x_test=sc.transform(x_test)

    # Xgboost parameters hard coded after grid-search cross validation
    booster=xgb(n_estimators=200,random_state=4,gamma=0.2,max_depth=6,learning_rate=0.1,
                colsample_bytree=0.7
            )

    # Fit model make predictions on test set and output the metrics
    booster.fit(x_train,y_train)
    pickle.dump(booster,OUTFILE)
    OUTFILE.close()

    # Validate model is predicting
    y_preds = booster.predict(x_test)
    for i in y_preds:
        print("$", round(i, 2), "/ night")

    return y_preds

def main():
    df = pd.read_csv("./data/airbnb_historical_listings.csv")
    # print("[INFO] Begin model training...")
    results = gradient_boosting(df)

if __name__ == "__main__":
    main()
