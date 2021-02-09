- [Project](#project)
  - [Objectives](#objectives)
  - [Input Data](#input-data)
    - [Preparation](#preparation)
      - [Features](#features)
    - [Data cleansing and registration](#data-cleansing-and-registration)
    - [Compute](#compute)
  - [AutoML run](#automl-run)
    - [Choosing primary metric](#choosing-primary-metric)
    - [Featurization](#featurization)
    - [Register model](#register-model)
  - [Hyperdrive run](#hyperdrive-run)
- [Resources](#resources)
- [Tmp links](#tmp-links)
- [Trash](#trash)
  - [Encoding](#encoding)
# Project

## Objectives

The purpose of the project is to predict the popularity of a product based on the the set of [features](#features).
The popularity index is based on the number of views (scans) of the product in different countries.

To simplify and since It's impossible to deduct cltural preferences from available data, I will concentrate on products which are sold in France.

## Input Data

### Preparation

The initial input dataset is extracted from the open source [data base](https://world.openfoodfacts.org/data) dedicated to food products [Open Food Facts](https://world.openfoodfacts.org/)
The Open Food Facts database is available under the [Open Database License](https://opendatacommons.org/licenses/dbcl/1-0/).

#### Features

One product database contains more than [150 fields](https://static.openfoodfacts.org/data/data-fields.txt). After each field relevance analysis (from the consumer point of view), here is a list of fields I chose to be transformed into features:

1. *categories_hierarchy*: array of categories the product belongs to,
2. *nutriscore_grade* : int, the [nutriscore grade](https://world.openfoodfacts.org/nutriscore),
3. *additives_n* : int, number of food additives,
4. *serving_quantity*: int, serving size in g,
6. *allergens_hierarchy*, *allergens_tags*: array of allergens tags
7. *nutrient_levels* object describing the levels:
```js
    {
      salt:"low|moderate|hight",
      sugars:"low|moderate|hight",
      saturated-fat:"low|moderate|hight",
      fat:"low|moderate|hight"
    }
```
8. *ingredients_analysis_tags*: the list of tags summarizing the ingredients (for example "vegan")
9. *brands_tags*: the list of manufacturer brands
10. *packagings*: array of { shape: string, material: string}


The data is filtered to contain only products sold in France ('countries_hierarchy' contains "en:france").

The initial features contained in the database are transformed using MongpDB [pipeline](./eda/products-reduce-dataset-pipeline.txt).

Resulting features:

1. *category_1* - categorical, nominal: the main category of the product
2. *category_2* - categorical, nominal: secondary category of the product
3. *vegan* - binary: 1 - yes or maybe, 0 - no or unknown (derived from 'ingredients_analysis_tags')
4. *vegetarian* - binary: 1 - yes or maybe, 0 - no or unknown  (derived from 'ingredients_analysis_tags')
5. *palm_oil* - binary: 1 - contains or may contain palm oil, 0 - palm-oil free or unknown (derived from 'ingredients_analysis_tags')
6. *brand* - categorical, nominal: the main brand (derived from 'brands_tags')
7. *additives_n* - integer: number of additives (copy of 'additives_n')
8. *salt_level* - categorical, ordinal: "low", "moderate", "hight" (derived from 'nutrient_levels')
9. *sugars_level* - categorical, ordinal: "low", "moderate", "hight" (derived from 'nutrient_levels')
10. *saturated_fat_level* - categorical, ordinal: "low", "moderate", "hight" (derived from 'nutrient_levels')
11. *fat_level* - categorical, ordinal: "low", "moderate", "hight" (derived from 'nutrient_levels')
12. *packaging_shape* - categorical, nominal: packaging shape (derived from 'packagings')
13. *packaging_material* - categorical: packaging material (derived from 'packagings')
14. *serving_quantity* - integer, serving size in g (copy of 'serving_quantity')
15. *nutriscore_grade* - categorical, the nutriscore (copy of 'nutriscore_grade')
16. *allergens_n* - number of allergens present in the product (derived from 'allergens_hierarchy')
17. *popularity_key* - popularity of the product based on the number of views (scans).

The resulting collection is exported with `mongpexport` utility.
`mongoexport` utility exports collection in [json lines](https://jsonlines.org/) format which is directly supported by `TabularDatasetFactory.from_json_lines_files`.
```
mongoexport -v --limit 10000 --collection=foods-features  --db=off --out=eda\foods-features-v3.json
```
### Data cleansing and registration

I perform the same data cleansing for the AutoMl run and for the HyperDrive run to have the most comparable results.

The cleansing is applied on the data exported from the database. The [cleansing script](./scripts/cleansing.py) uses pandas and SKLearn to transform features into numerical values:
* categorical features transformation: hashing, hot encoding
* missing values replacement
* binary values transformation
  
The resulting dataset is registered in the ML workspace. 
```py
from scripts.cleansing import clean_data
import pandas as pd

def get_cleaned_dataset():
    found = False
    ds_key = "openfoodfacts"
    description_text = "Data extracted from OpenFoodFacts open source database."

    if ds_key in ws.datasets.keys(): 
        found = True
        ds_cleaned = ws.datasets[ds_key] 

    # Otherwise, create it from the file
    if not found:
        #Reading a json lines file into a DataFrame
        data = pd.read_json('./eda/foods-features-v3.json', lines=True)
        # DataFrame with cleaned data
        data_cleaned = clean_data(data)
        exported_df = 'cleaned-openfoodfacts.parquet'
        cleaned_data.to_parquet(exported_df);
        # Register Dataset in Workspace using experimental funcionality to upload and register pandas dataframe at once
        ds_cleaned = TabularDatasetFactory.register_pandas_dataframe(dataframe=cleaned_data,
                                                                     target=(ws.get_default_datastore(), exported_df),
                                                                     name=ds_key, description=description_text,
                                                                     show_progress=True)
    return ds_cleaned
```

### Compute

To perform AutoML run and Hyperdrive run I create a compute cluster:
```py
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

#cluster_name = "cluster-nd-capstone"
cluster_name = "auto-ml"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                           #vm_priority = 'lowpriority', # optional
                                                           max_nodes=4, min_nodes=1)
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
```


## AutoML run

### Choosing primary metric

Predicting a populatiry_key is a [Regression](https://en.wikipedia.org/wiki/Linear_regression) task.

According to Azure [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#primary-metrics-for-regression-scenarios):
> metrics like `r2_score` and `spearman_correlation` can better represent the quality of model when the scale of the value-to-predict covers many orders of magnitude.

The 'popularity_key' value minimum value is 1.200000e+01, maximum value is 1.199999e+11.

I'll be using `r2_score` metric, since it is supported by AutoML and by [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/ensemble.html#regression) which I use for HyperDrive run. 

### Featurization

AutoML run performs automatic [featurization](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-configure-auto-features.md#automatic-featurization) (with `featurization ='auto'` parameter): > Featurization includes automated feature engineering (when "featurization": 'auto') and scaling and normalization, which then impacts the selected algorithm and its hyperparameter values.

The result of featurization can be retrieved and inspected from the best model run:
```python
best_run, fitted_model = auto_ml_run.get_output()
summary = fitted_model.named_steps['datatransformer'].get_featurization_summary()
```

To see the result of the normalization scaling/normalization and the details of the selected algorithm with its hyperparameter values, we can use `fitted_model.steps` and the [helper function](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features#scaling-and-normalization) provided in a Azure tutorial.


### Register model

[Framework](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model.framework?view=azure-ml-py) constants simplify deployment for some popular frameworks. Use the framework constants in the Model class when registering or searching for models.

## Hyperdrive run

From the AutoML run we can see that the 3 best perming models are:
1. with % 
2. with %
3. with %

We can tune one of these models using Hyperdrive to choose the optimal hyperparameter values.
The [hyperdrive-run](./hyperdrive-run.ipynb) notebook 



# Resources

* [Tutorial: Use automated machine learning to predict taxi fares](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-auto-train-models)
* [ONNX models deployment examples](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx)

# Tmp links
https://hub.docker.com/_/microsoft-azureml-onnxruntimefamily?tab=description
https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#supported-models
https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/linear-regression
https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-auto-train-models
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-automlstep-in-pipelines


https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx
deploy with ONNX environment https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-model-register-and-deploy.ipynb

example deploying to IoT Edge https://github.com/Azure-Samples/onnxruntime-iot-edge/blob/master/AzureML-OpenVINO/AML-BYOC-ONNXRUNTIME-OpenVINO.ipynb

# Trash

15. *allergen_milk* - binary: 1 - present, 0 - no
16. *allergen_gluten* - binary: 1 - present, 0 - no
17. *allergen_soybeans* - binary: 1 - present, 0 - no
18. *allergen_eggs* - binary: 1 - present, 0 - no
19. *allergen_nuts* - binary: 1 - present, 0 - no
20. *allergen_celery* - binary: 1 - present, 0 - no
21. *allergen_mustard* - binary: 1 - present, 0 - no
22. *allergen_peanuts* - binary: 1 - present, 0 - no
23. *allergen_seafood* - binary: 1 - present, 0 - no (includes fish, crustaceans, molluscs)
24. *allergen_lupin* - binary: 1 - present, 0 - no

For allergens, only 10 top values (with the most products associates) are represented.

## Encoding

MultiLabelBinarizer
https://chrisalbon.com/machine_learning/preprocessing_structured_data/one-hot_encode_features_with_multiple_labels/

https://stackoverflow.com/questions/15181311/using-dictvectorizer-with-sklearn-decisiontreeclassifier

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html

https://towardsdatascience.com/beginners-guide-to-encoding-data-5515da7f56ab