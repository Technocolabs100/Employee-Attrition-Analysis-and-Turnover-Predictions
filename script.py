import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings('ignore', category=joblib.InconsistentVersionWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

model_path = r'D:\Mahine Learning\technocolab intern\Employee-Attrition-Analysis-and-Turnover-Predictions\resampled_data.pkl'
model = joblib.load(model_path)

def predict_attrition(data):
    predictions = model.predict(data)
    return predictions

dataset_path = r'D:\Mahine Learning\technocolab intern\Employee-Attrition-Analysis-and-Turnover-Predictions\df_data.csv'

data = pd.read_csv(dataset_path)

predictions = predict_attrition(data)

data['Predictions'] = predictions

plt.bar(['Stay', 'Leave'], data['Predictions'].value_counts())
plt.xlabel('Attrition Prediction')
plt.ylabel('Count')
plt.title('Predicted Attrition Distribution')
plt.show()
