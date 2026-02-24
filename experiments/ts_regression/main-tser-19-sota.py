from aeon.regression.convolution_based import RocketRegressor
from sklearn.metrics import root_mean_squared_error
from tsml.compose import SklearnToTsmlRegressor
from tsml.datasets import load_minimal_gas_prices
from utils.data_loader import load_from_tsfile_to_dataframe

from tsml_eval.publications.y2023.tser_archive_expansion import _set_tser_exp_regressor
from tsml_eval.utils.estimator_validation import is_sklearn_regressor


# data_name = ['AppliancesEnergy', 'HouseholdPowerConsumption1', 'HouseholdPowerConsumption2', 'BenzeneConcentration',
#              'BeijingPM25Quality', 'BeijingPM10Quality', 'LiveFuelMoistureContent', 'FloodModeling1',
#              'FloodModeling2',
#              'FloodModeling3', 'AustraliaRainfall', 'PPGDalia', 'IEEEPPG', 'BIDMCRR', 'BIDMCHR', 'BIDMCSpO2',
#              'NewsHeadlineSentiment',
#              'NewsTitleSentiment', 'Covid3Month']


data_name = 'AppliancesEnergy'

regressors=['MultiROCKET', 'InceptionE', 'TSF', 'Ridge', 'RotF', 'DrCIF', 'FreshPRINCE', 'CNN']

data_folder = 'data/monash-regression/'
train_file = data_folder + data_name + "_TRAIN.ts"
test_file = data_folder + data_name + "_TEST.ts"

X_train, y_train = load_from_tsfile_to_dataframe(train_file)
X_test, y_test = load_from_tsfile_to_dataframe(test_file)

X_train = X_train.iloc[1:, :]
print(X_train)
print(y_train)
print(type(X_train), type(y_train))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


rmse = []
for regressor_name in regressors:
    # Select a regressor by name, see set_tser_exp_regressor.py for options
    regressor = _set_tser_exp_regressor(regressor_name, random_state=0)

    # if it is a sklearn regressor, wrap it to work with time series data
    if is_sklearn_regressor(regressor):
        regressor = SklearnToTsmlRegressor(
            regressor=regressor, concatenate_channels=True, random_state=0
        )

    # fit and predict
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    rmse.append(root_mean_squared_error(y_test, y_pred))
print(rmse)

