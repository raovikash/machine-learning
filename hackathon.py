# importing required libraries
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# loading the dataset
all_data = pd.read_csv('train.csv')

# converting the columns to category
def convert_columns_as_category(columns, data):
    for column in columns:
        data[column] = data[column].astype('category')

# dropping the columns which are not required
all_data = all_data.drop(['registered_color', 'vehicle_make'], axis=1)

categorical_columns = ['car_variant', 'vehicle_fuel_type', 'vehicle_model', 'city', 'accidental_vehicle']
convert_columns_as_category(categorical_columns, all_data)

# train test split on all_data


