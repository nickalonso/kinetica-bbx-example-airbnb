import pickle
import pandas as pd

#Import trained XGboost model and trained scaler for data normalization
BOOSTER =pickle.load(open("model", "rb"))
SCALER = pickle.load(open("scaler", "rb"))

def estimate_price(bedrooms,bathrooms,sqft,accommodates,distance,cleaning):
    """This function calls our saved XGboost model and predicts based on received data.
    Set default values first which can be tuned based on table averages and overlay values with
    user input once provided"""
    defaults = {
        'accommodates':[3],
        'bedrooms':[1],
        'bathrooms':[1],
        'cleaning_fee':[0.0],
        'distance':[4.0],
        'size':[400]
    }

    # Overlay values from user input
    df=pd.DataFrame.from_dict(defaults)
    df['bedrooms']=bedrooms
    df['bathrooms']=bathrooms
    df['size']=sqft
    df['accommodates']=accommodates
    df['distance']=distance
    df['cleaning_fee']=cleaning

    # Apply scaling and predict with trained xgboost model
    x_test=df
    x_test=SCALER.transform(x_test)
    preds=BOOSTER.predict(x_test)
    return preds

def blackbox_function_airbnb(inMap):
    """Features obtained from user within KML are passed to method as a dictionary (inMap) and
    predicted price is returned to KML as a dictionary (outMap)"""
    bedrooms = int(inMap['bedrooms'])
    bathrooms = int(inMap['bathrooms'])
    sqft = int(inMap['size'])
    accommodates = int(inMap['accommodates'])
    distance = int(inMap['distance'])
    cleaning_fee = float(inMap['cleaning_fee'])

    # Send received feature set to our function that calls our pre-trained model
    prices=estimate_price(bedrooms,bathrooms,sqft,accommodates,distance,cleaning_fee)
    results=round(prices[0],2)

    # Predicted price returned to KML as a dictionary
    outMap={"price": results}
    return outMap