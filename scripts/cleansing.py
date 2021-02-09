import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

def feature_hashing(data, col_name, features_number):
	#create feature hasher for column values
	fh = FeatureHasher(n_features=features_number, input_type='string')
	col_values = data[col_name].tolist()
	hashed_matrix = fh.transform(col_values)
	#create new column names for hashed features
	rng = range(0, features_number)
	new_cols =  [col_name + '_' + str(i) for i in rng]
	df_to_join = pd.DataFrame(hashed_matrix.toarray())
	df_to_join.columns = new_cols[:features_number]
	#join hashed feature matrix to the dataframe
	data = data.join(pd.DataFrame(df_to_join))
	# drop original column
	data.drop(col_name, inplace=True, axis=1)
	return data

""" Cleans the data. Returns cleaned data. """
def clean_data(data):
	# Ordinal categorical features
	levels = {"unknown": 0, "low":1, "moderate":2, "high":3}
	nutriscores = {"a":1, "b":2, "c":3, "d":4, "e":5 }


	# Clean and one hot encode data
	x_df = data.fillna({'packaging_shape': "unknown"})
	x_df.drop("_id", inplace=True, axis=1)
	
	pd.set_option('display.max_columns', None)

	# fill in all null values for string objects with unknown
	str_cols = x_df.columns[x_df.dtypes==object]
	x_df[str_cols] = x_df[str_cols].fillna('unknown')
	
	
	# fill 'serving_quantity_g' with avarage value of the column
	x_df["serving_quantity_g"].fillna(value = x_df["serving_quantity_g"].mean(), inplace=True)
	
	# fill all NaN numeric columns with 0
	x_df = x_df.fillna(0)
	
	x_df["salt_level"] = x_df.salt_level.map(levels)
	x_df["sugar_level"] = x_df.sugar_level.map(levels)
	x_df["fat_level"] = x_df.fat_level.map(levels)
	x_df["saturated_fat_level"] = x_df.saturated_fat_level.map(levels)
	
	x_df["nutriscore_grade"] = x_df.nutriscore_grade.map(nutriscores)
	
	# hot encoding for nominal categorical features
	packaging_materials = pd.get_dummies(x_df.packaging_material, prefix="pack_mat")
	x_df.drop("packaging_material", inplace=True, axis=1)
	x_df = x_df.join(packaging_materials)
	
	packaging_shapes = pd.get_dummies(x_df.packaging_shape, prefix="pack_sh")
	x_df.drop("packaging_shape", inplace=True, axis=1)
	x_df = x_df.join(packaging_shapes)
	
	# binary columns
	x_df["vegan"] = x_df.vegan.apply(lambda s: 1 if s else 0)
	x_df["vegeterian"] = x_df.vegeterian.apply(lambda s: 1 if s else 0)
	x_df["palm_oil"] = x_df.palm_oil.apply(lambda s: 1 if s else 0)
	
	
	# create brand feature hashing matrix and append it to the frame
	x_df = feature_hashing(x_df, 'brand', 1024) 
	x_df = feature_hashing(x_df, 'category_1', 128)
	x_df = feature_hashing(x_df, 'category_2', 128)	

	return x_df
