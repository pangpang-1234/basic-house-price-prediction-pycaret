# House price prediction using Pycaret Regression
## Connect with me

<a href="https://www.linkedin.com/in/piyapadech/">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
</a>

## About
I use dataset from house price prediction kaggle competition datasets to explore Pycaret features. In this project, Pycaret Regression was applied to predict house price from previous data. I straightforward to use Pycaret, so there are no preprocess or explore data in this project. You can find Pycaret tutorials here <a href='https://pycaret.gitbook.io/docs/get-started/tutorials'>Pycaret Tutorials</a>

## Setup
Firstly, install pycaret full package with this command
```
pip install pycaret[full]
```
Then import pycaret regression and other necessary libraries  
```
from pycaret.regression import *
import pandas as pd
```
## Prepare Data
Prepare path and create dataframe
```
root_path = '/content/drive/Shareddrives/SuperAI/Kaggle/home-data-for-ml-course/'
data = pd.read_csv('/content/drive/Shareddrives/SuperAI/Kaggle/home-data-for-ml-course/train.csv')
test_data = pd.read_csv('/content/drive/Shareddrives/SuperAI/Kaggle/home-data-for-ml-course/test.csv')
```
Setup data before finding best model
1. data = train dataset
2. target = what we want to predict
3. ignore_features = features that Pycaret will not use to train model

find more detail here <a href="https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Intermediate%20-%20REG102.ipynb">Pycaret Regression Intermediate</a>
```
demo = setup(data = data, target = 'SalePrice', 
                   ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],normalize = True,
                   transformation= True, transformation_method = 'yeo-johnson', 
                   transform_target = True, remove_outliers= True,
                   remove_multicollinearity = True,
                   ignore_low_variance = True, combine_rare_levels = True) 
```
Compare to find best model (if you don't have much time, you can delete turbo=False to accelerate compare models)
``` 
best_model = compare_models(sort='rmse', turbo=False)
```
## Predict
Predict house price model need 2 arguments 1.estimator = model 2.data = test dataset
```
out = predict_model(estimator=loaded_model, data=test_data)
out
```
## Save load and evaluate model
Save model need 2 arguments 1.model 2.path to save, for example
```
save_model(best_model,'/content/b_model')
```
load model need 1 argument that is model path, for excample
```
loaded_model=load_model('/content/b_model')
```
evaluate need 1 argument that is model, for example
```
evaluate_model(best_model)
```

Thanks to Nontapath Taspan for sharing Pycaret Regression knowledge
