from sklearn.ensemble import GradientBoostingRegressor
import math
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory

""" Splits the data to data used for training and labels to predict. Returns a tuple (data, labels)"""
def split_train_label_data(x_df):
    # column to predict
    y_df = x_df.pop("ERP")

    return (x_df, y_df)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    parser.add_argument('--n_estimators', type=int, default=100, help="The number of boosting stages to perform")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate shrinks the contribution of each tree by learning_rate")

    args = parser.parse_args()

    run = Run.get_context()
    # Get the dataset from run inputs
    #ds = run.input_datasets['dataset']

    ws = run.experiment.workspace
    # get the input dataset by ID
    ds = Dataset.get_by_id(ws, id=args.input_data)

    df = ds.to_pandas_dataframe()
    erp_mean = df['ERP'].mean()
    x, y = split_train_label_data(df)

    # Split data into train and test sets: 20% of the dataset to include in the test split.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
 
    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("learning_rate:", np.float(args.learning_rate))

    model = GradientBoostingRegressor(n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                                    max_depth=1, random_state=0, loss='huber').fit(x_train, y_train)
    # MSE 
    mse = mean_squared_error(y_test, model.predict(x_test))

    # normalized_root_mean_squared_error => to be comparabe with Azure results
    metric = math.sqrt(mse)/erp_mean

    run.log("normalized_root_mean_squared_error", np.float(metric))
    # Metric reported is 'r2_score' => metric to optimize
    run.log("r2_score", np.float(model.score(x_test, y_test)))

    os.makedirs('outputs', exist_ok=True)
    # Save the model into run history
    joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
