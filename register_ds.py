import pandas as pd
import zipfile
from scripts.cleansing import clean_data
from azureml.data.dataset_factory import TabularDatasetFactory


def get_cleaned_dataset(ws):
    found = False
    ds_key = "machine-cpu"
    description_text = "CPU performance dataset (UCI)."

    if ds_key in ws.datasets.keys(): 
        found = True
        ds_cleaned = ws.datasets[ds_key] 

    # Otherwise, create it from the file
    if not found:

        with zipfile.ZipFile("./data/machine.zip","r") as zip_ref:
            zip_ref.extractall("data")

        #Reading a json lines file into a DataFrame
        data = pd.read_csv('./data/machine.csv')
        # DataFrame with cleaned data
        cleaned_data = clean_data(data)
        exported_df = 'cleaned-machine-cpu.parquet'
        cleaned_data.to_parquet(exported_df)
        # Register Dataset in Workspace using experimental funcionality to upload and register pandas dataframe at once
        ds_cleaned = TabularDatasetFactory.register_pandas_dataframe(dataframe=cleaned_data,
                                                                     target=(ws.get_default_datastore(), exported_df),
                                                                     name=ds_key, description=description_text,
                                                                     show_progress=True)
    return ds_cleaned
