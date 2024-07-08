# importing required libraries
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import mplcursors
from catboost import CatBoostRegressor

categorical_columns = ['vehicle_make', 'vehicle_model', 'city', 'car_variant', 'vehicle_fuel_type']
columns_to_be_dropped = ['month_of_vehicle_manufacture', 'year_of_vehicle_manufacture', 'registered_color',
                         'accidental_vehicle', 'Odometer_Reading_Present']

# return integer month, from string, default is 6
def month_to_number(month):
    if pd.isna(month):
        return 6
    try:
        # Try to convert directly to an integer
        month_number = int(month)
        if 1 <= month_number <= 12:
            return month_number
    except ValueError:
        # If month is a string, convert it to lower case and map to month number
        month = month.lower()
        month_dict = {
            'january': 1, 'february': 2, 'march': 3,
            'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9,
            'october': 10, 'november': 11, 'december': 12
        }
        return month_dict.get(month, 6)  # Return 6 if month is not recognized
    return 6

# Function to calculate count of months since manufacting data(year, month)
def calculate_months_since_manufactured(row, current_year, current_month):
    year, month = row['year_of_vehicle_manufacture'], row['month_of_vehicle_manufacture']
    return (current_year - year) * 12 + (current_month - month)
    
# Fuction to mark columns as category type
def convert_columns_as_category(columns, data):
    for column in columns:
        data[column] = data[column].astype('category')

# Convert values to uppercase
def convert_values_to_uppercase(columns, data):
    for column in columns:
        data[column] = data[column].str.strip()
        data[column] = data[column].str.upper()

# Replace text which is matching with a pattern
def replace_text(pattern, data, column):
    data[column] = data[column].str.replace(pattern, '', regex=True)

def train_model(df, model):
    train_data = df.sample(frac = 0.9)
    test_data = df.drop(train_data.index)
    train_inputs = train_data.drop(columns=['car_valuation'], axis=1)
    train_output = train_data['car_valuation']
    test_inputs = test_data.drop(columns=['car_valuation'], axis=1)
    test_output = test_data['car_valuation']
    model.fit(train_inputs, train_output, cat_features=categorical_columns)
    # mean absolute error on train data
    train_prediction = model.predict(train_inputs)
    train_error = mean_absolute_error(train_prediction, train_output)
    # mean absolute error on test data
    test_predicted_output = model.predict(test_inputs)
    test_error = mean_absolute_error(test_predicted_output, test_output)
    print("test_error= ",test_error," train_error= ", train_error)
    # Plot Feature Importance
    # plot_feature_importance(model, train_inputs.columns)
    return test_error, train_error

# Training
def prune_data_and_train_model(data, model):
    data = prune_data(data)
    data = data.drop(columns=['id'], axis=1)
    train_model(data, model)

# Prune data
def prune_data(data):
    # set month number, considering string, int, nan cases 
    data['month_of_vehicle_manufacture'] = data['month_of_vehicle_manufacture'].apply(month_to_number)
    # add months_since_manufactured column
    data['months_since_manufactured'] = data.apply(calculate_months_since_manufactured, axis=1, current_year=datetime.now().year, current_month=datetime.now().month)
    # remove redundant features
    data = data.drop(columns=columns_to_be_dropped, axis=1)
    # convert values to uppercase
    convert_values_to_uppercase(categorical_columns, data)
    # remove yyyy-yyyy or yyyy from vehicle_model
    pattern = r'\b\d{4}-\d{4}\b|\b\d{4}\b'
    replace_text(pattern, data, 'vehicle_model')
    # mark columns as category type
    convert_columns_as_category(categorical_columns, data)
    return data

def separate_on_the_basis_of_vehicle_maker(data):
    premium_car_makers = ['JEEP', 'BMW', 'JAGUAR', 'ISUZU', 'AUDI', 'TOYOTA ETIOS', 'MG', 'MG MOTOR INDIA PVT LTD',
                      'TOYOTA KIRLOSKAR MOTORS LTD', 'MERCEDES', 'KIA', 'MERCEDES-BENZ']
    data_premium_cars = data[data['vehicle_make'].isin(premium_car_makers)]
    data_other_cars = data[~data['vehicle_make'].isin(premium_car_makers)]
    return data_premium_cars, data_other_cars
    
# Training
def prune_data_and_train_model(data):
    data = prune_data(data)
    data = data.drop(columns=['id'], axis=1)
    data_premium_cars, data_other_cars = separate_on_the_basis_of_vehicle_maker(data)
    model_for_premium_cars = CatBoostRegressor(iterations=1000, learning_rate=0.5, depth=6, loss_function='MAE', verbose=500)
    model_for_other_cars = CatBoostRegressor(iterations=1000, learning_rate=0.5, depth=6, loss_function='MAE', verbose=500)
    test_error_premium_cars, train_error_premium_cars = train_model(data_premium_cars, model_for_premium_cars)
    print("Model for premium cars trained, test_error_premium_cars", test_error_premium_cars, 
          " train_error_premium_cars", train_error_premium_cars)
    test_error_other_cars, train_error_other_cars = train_model(data_other_cars, model_for_other_cars)
    print("Model for other cars trained, test_error_other_cars", test_error_other_cars, 
          " train_error_other_cars", train_error_other_cars)
    return model_for_premium_cars, model_for_other_cars

# return ids and predicted outcome using model
def run_model(data, model):
    ids = data['id']
    data = data.drop(columns=['id'], axis=1)
    predicted_output = model.predict(data)
    return ids, predicted_output

# predict outcomes and save in file
def predict_output_separately_and_save(data_premium_cars, model_for_premium_cars, data_other_cars, model_for_other_cars):
    ids_premium_cars, prediction_for_premium_cars = run_model(data_premium_cars, model_for_premium_cars)
    ids_other_cars, prediction_for_other_cars = run_model(data_other_cars, model_for_other_cars)
    ids = []
    for id_more_liked in ids_premium_cars:
        ids.append(id_more_liked)
    for id_less_liked in ids_other_cars:
        ids.append(id_less_liked)
    
    car_valuations = []
    for valuation in prediction_for_premium_cars:
        car_valuations.append(int(valuation))
    for valuation in prediction_for_other_cars:
        car_valuations.append(int(valuation))
    print(len(car_valuations))
    
    df = pd.DataFrame({
        'id': ids,
        'car_valuation': car_valuations
    })
    df.to_csv('car_valuation_data.csv', index=False)

def predict_output(data, model_for_premium_cars, model_for_other_cars):
    data = prune_data(data)
    data_premium_cars, data_other_cars = separate_on_the_basis_of_vehicle_maker(data)
    predict_output_separately_and_save(data_premium_cars, model_for_premium_cars, 
                                       data_other_cars, model_for_other_cars)


# Load data
data = pd.read_csv('train.csv')
model_for_premium_cars, model_for_other_cars = prune_data_and_train_model(data)

# Load test data
data = pd.read_csv('test.csv')
predict_output(data, model_for_premium_cars, model_for_other_cars)
