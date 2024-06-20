import pandas as pd
import joblib


model = joblib.load('resampled_data.pkl.pkl')


def predict_attrition(data):
    predictions = model.predict(data)
    return predictions


data = pd.read_csv('Attrition Dataset\WA_Fn-UseC_-HR-Employee-Attrition.csv')
predictions = predict_attrition(data)


data['Predictions'] = predictions
