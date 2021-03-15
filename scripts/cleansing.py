import numpy as np
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler

""" Cleans the data. Returns cleaned data. """
def clean_data(data):
	# Clean and one hot encode data
	x_df = data.drop("model_name", axis=1)
	
	# hot encoding for nominal categorical features
	vendors = pd.get_dummies(x_df.vendor, prefix="vendor")
	x_df.drop("vendor", inplace=True, axis=1)
	x_df = x_df.join(vendors)

	# normalize data
	#min_max_scaler = MinMaxScaler()
	#cols_names = ['MYCT','MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP']
	#norm_cols = x_df[cols_names]
	#scaled_df = min_max_scaler.fit_transform(norm_cols)
	#x_df[cols_names] = scaled_df
	
	print(f"Rows*columns={x_df.shape[0]}*{x_df.shape[1]}")

	return x_df
