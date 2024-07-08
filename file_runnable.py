import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from datetime import datetime

categorical_columns = ['vehicle_make', 'vehicle_model', 'city']
columns_to_be_upper_cased = ['vehicle_make', 'vehicle_model', 'city']
columns_to_be_dropped = ['month_of_vehicle_manufacture', 'year_of_vehicle_manufacture', 'registered_color',
                         'accidental_vehicle', 'Odometer_Reading_Present', 'car_variant', 'vehicle_fuel_type', 'odometer_reading']

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
    return (current_year - year)
    # return (current_year - year) * 12 + (current_month - month)
    
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

# Prune data
def prune_data(data):
    # set month number, considering string, int, nan cases 
    data['month_of_vehicle_manufacture'] = data['month_of_vehicle_manufacture'].apply(month_to_number)
    # add months_since_manufactured column
    data['months_since_manufactured'] = data.apply(calculate_months_since_manufactured, axis=1, current_year=datetime.now().year, current_month=datetime.now().month)
    # remove redundant features
    data = data.drop(columns=columns_to_be_dropped, axis=1)
    # convert values to uppercase
    convert_values_to_uppercase(columns_to_be_upper_cased, data)
    # remove yyyy-yyyy or yyyy from vehicle_model
    pattern = r'\b\d{4}-\d{4}\b|\b\d{4}\b'
    replace_text(pattern, data, 'vehicle_model')
    # mark columns as category type
    # convert_columns_as_category(categorical_columns, data)
    return data

# Train the model
def train_model(df, model):
    train_data = df.sample(frac = 0.8)
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
    print("test_error",test_error,"train_error", train_error)
    # Plot Feature Importance
    # plot_feature_importance(model, train_inputs.columns)
    return test_error, train_error

# return ids and predicted outcome using model
def run_model(data, model):
    predicted_output = model.predict(data)
    return predicted_output

# predict outcomes and save in file
def predict_output_and_save(ids, data, model, file_name_suffix):
    car_valuations = run_model(data, model)
    car_valuations = car_valuations.astype(int)
    df = pd.DataFrame({
        'id': ids,
        'car_valuation': car_valuations
    })
    df.to_csv('/Users/vikash.yadav/Documents/data_science_hackathon/' + file_name_suffix + '_car_valuation_data.csv', index=False)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data = prune_data(train_data)
test_data = prune_data(test_data)


# def replace_missing_values(data, column, value):
#     data[column] = data[column].fillna(value)

# replace_missing_values(train_data, 'vehicle_make', 'UNKNOWN')
# replace_missing_values(train_data, 'vehicle_model', 'UNKNOWN')
# replace_missing_values(train_data, 'city', 'UNKNOWN')
# replace_missing_values(train_data, 'odometer_reading', 100000)

# replace_missing_values(test_data, 'vehicle_make', 'UNKNOWN')
# replace_missing_values(test_data, 'vehicle_model', 'UNKNOWN')
# replace_missing_values(test_data, 'city', 'UNKNOWN')
# replace_missing_values(test_data, 'odometer_reading', 100000)
    

# write code to encode the categorical columns in train and test data using label encoder, and put -1 in case
# of unknown or missing values
# def encode_categorical_columns(train_data, test_data, categorical_columns):
#     for column in categorical_columns:
#         encoder = LabelEncoder()
#         # Fit and transform on the train data
#         # handle unknown values
#         train_data[column] = train_data[column].fillna('UNKNOWN')
#         train_data[column] = encoder.fit_transform(train_data[column])
#         # Transform on the test data using the same encoder
#         # handle values which were not there in train data
#         test_data[column] = test_data[column].fillna('UNKNOWN')
#         test_data[column] = encoder.transform(test_data[column])
        
#     return train_data, test_data

# encode_categorical_columns(train_data, test_data, categorical_columns)
# train_data.to_csv('/Users/vikash.yadav/Documents/data_science_hackathon/train_label_encoded.csv', index=False)
# test_data.to_csv('/Users/vikash.yadav/Documents/data_science_hackathon/test_label_encoded.csv', index=False)
                                              
# encoder = ce.OneHotEncoder(handle_unknown='indicator', use_cat_names=True)
# # Encode the categorical columns in train and test data
# for column in categorical_columns:
#     # Fit and transform on the train data
#     transformed_train = encoder.fit_transform(train_data[column])
#     train_data = pd.concat([train_data.drop(columns=[column]), transformed_train], axis=1)
    
#     # Transform on the test data using the same encoder
#     transformed_test = encoder.transform(test_data[column])
#     test_data = pd.concat([test_data.drop(columns=[column]), transformed_test], axis=1)


# train_data.to_csv('/Users/vikash.yadav/Documents/data_science_hackathon/train_encoded.csv', index=False)
# test_data.to_csv('/Users/vikash.yadav/Documents/data_science_hackathon/test_encoded.csv', index=False)

# Using CatBoostRegressor
# model = CatBoostRegressor(iterations=5000, learning_rate=0.1, depth=6, loss_function='MAE', verbose=500)
# model = XGBRegressor(eval_metric='mae', booster='dart', max_depth=100, learning_rate=0.525, n_estimators=500, random_state=42)
model_map = {
    'CatBoostRegressor': CatBoostRegressor(iterations=3000, learning_rate=0.05, depth=6, border_count=100, l2_leaf_reg=3,  loss_function='MAE', bagging_temperature=10, verbose=1000)
    # 'RandomForestRegressor': RandomForestRegressor(),
    # 'AdaBoostRegressor': AdaBoostRegressor(),
    # 'GradientBoostingRegressor': GradientBoostingRegressor(),
    # 'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=1000)
    # 'LinearRegression': LinearRegression(),
    # 'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.2)
    # 'BayesianRidge': BayesianRidge()
}

ids = test_data['id']
test_data = test_data.drop(columns=['id'], axis=1)
train_data = train_data.drop(columns=['id'], axis=1)
for model_name, model in model_map.items():
    print("Training model: ", model_name)
    try:
        train_model(train_data, model)
        print("Training completed_for_model: ", model_name)
        predict_output_and_save(ids, test_data, model, model_name)
    except Exception as e:
        print("Excpetion while training model: ", model_name, " Exception: ", e)