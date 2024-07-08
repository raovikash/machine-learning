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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

# Train the model
def train_model(df, model):
    train_data = df.sample(frac = 0.8)
    test_data = df.drop(train_data.index)
    train_inputs = train_data.drop(columns=['car_valuation'], axis=1)
    train_output = train_data['car_valuation']
    test_inputs = test_data.drop(columns=['car_valuation'], axis=1)
    test_output = test_data['car_valuation']
    # model.fit(train_inputs, train_output, cat_features=categorical_columns)
    model.fit(train_inputs, train_output)
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

# predict outcomes and save in file
def predict_output_and_save(ids, data, model, file_name_suffix):
    car_valuations = model.predict(data)
    car_valuations = car_valuations.astype(int)
    df = pd.DataFrame({
        'id': ids,
        'car_valuation': car_valuations
    })
    df.to_csv('/Users/vikash.yadav/Documents/data_science_hackathon/' + file_name_suffix + '_car_valuation_data.csv', index=False)

train_data = pd.read_csv('label_encoded_train.csv')
test_data = pd.read_csv('label_encoded_test.csv')

ids = test_data['id']
test_data = test_data.drop(columns=['id'], axis=1)
train_data = train_data.drop(columns=['id'], axis=1)

model_map = {
    # 'CatBoostRegressor': CatBoostRegressor(iterations=500, learning_rate=0.5, depth=3, loss_function='MAE', verbose=1000),
    'RandomForestRegressor': RandomForestRegressor(),
    # 'XGBRegressor': XGBRegressor(eval_metric='mae', enable_categorical=True, booster='dart',
    #                             max_depth=12, learning_rate=0.05, n_estimators=12, random_state=42),
    # 'AdaBoostRegressor': AdaBoostRegressor(),
    # 'GradientBoostingRegressor': GradientBoostingRegressor(),
    # 'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=20),
    # 'LinearRegression': LinearRegression(),
    # 'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.2),
    # 'BayesianRidge': BayesianRidge(),
    # 'Ridge': Ridge(1),
}

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': [5, 6, 7]
}

# # gbr =  GradientBoostingRegressor()
# gbr_with_best_params = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, max_features=6, min_samples_leaf=10,
#                                                   min_samples_split=5, n_estimators=1000, subsample=0.8)


# # grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
# train_model(train_data, gbr_with_best_params)
# # Best hyperparameters
# # best_params = grid_search.best_params_
# # print(f'Best hyperparameters: {best_params}')
# predict_output_and_save(ids, test_data, gbr_with_best_params, 'gbr_with_best_params')


rf = RandomForestRegressor(n_estimators=200, 
                           min_samples_split=2, 
                           min_samples_leaf=1, 
                           max_features='log2', 
                           criterion='absolute_error',
                           random_state=42)

train_model(train_data, rf)
predict_output_and_save(ids, test_data, rf, 'rf')


# for model_name, model in model_map.items():
#     print("Training model: ", model_name)
#     # for param in param_grid:
#     try:
#         # print("Training model: ", model_name, " with param: ", param)
#         # model = Ridge(alpha=param)
#         # grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
#         train_model(train_data, model)
#         print("Training completed_for_model: ", model_name)
#         predict_output_and_save(ids, test_data, model, model_name)
#     except Exception as e:
#         print("Excpetion while training model: ", model_name, " Exception: ", e)