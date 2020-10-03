#%matplotlib inline

import os
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from time import gmtime, strftime
from sklearn.datasets import load_boston
import sklearn.model_selection

import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# This is an object that represents the SageMaker session that we are currently operating in. This
# object contains some useful information that we will need to access later such as our region.
session = sagemaker.Session()

# This is an object that represents the IAM role that we are currently assigned. When we construct
# and launch the training job later we will need to tell it what IAM role it should have. Since our
# use case is relatively simple we will simply assign the training job the role we currently have.
role = get_execution_role()


## Step 1: Downloading the data
boston = load_boston()


## Step 2: Preparing and splitting the data
X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

# We split the dataset into 2/3 training and 1/3 testing sets.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_bos_pd, Y_bos_pd, test_size=0.33)

# Then we split the training set further into 2/3 training and 1/3 validation sets.
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.33)


## Step 3: Uploading the training and validation files to S3
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

prefix = 'boston-update-endpoints'

# Upload to S3
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)


## Step 4 (A): Train the XGBoost model
xgb_container = get_image_uri(session.boto_region_name, 'xgboost')

# Now that we know which container to use, we can construct the estimator object.
xgb = sagemaker.estimator.Estimator(xgb_container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='text/csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='text/csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})


## Step 6 (A): Deploy the trained model
xgb_model_name = "boston-update-xgboost-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

xgb_primary_container = {"Image": xgb_container, "ModelDataUrl": xgb.model_data}

# And lastly we construct the SageMaker model
xgb_model_info = session.sagemaker_client.create_model(ModelName=xgb_model_name,
                                                       ExecutionRoleArn=role,
                                                       PrimaryContainer=xgb_primary_container)

# Create endpoint configuration
xgb_endpoint_config_name = "boston-update-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

xgb_endpoint_config_info = session.sagemaker_client.create_endpoint_config(EndpointConfigName=xgb_endpoint_config_name,
                                                                           ProductionVariants=[{
                                                                               "InstanceType": "ml.m4.xlarge",
                                                                               "InitialVariantWeight": 1,
                                                                               "InitialInstanceCount": 1,
                                                                               "ModelName": xgb_model_name,
                                                                               "VariantName": "XGB-Model"
                                                                           }])

# Deploy the endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

endpoint_info = session.sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=xgb_endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)

## Step 7 (A): Use the model
response = session.sagemaker_runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                                            ContentType="text/csv",
                                                            Body=','.join(map(str, X_test.values[0])))

print(response)
result = response['Body'].read().decode('utf-8')
print(result)
print(Y_test.values[0])

# Shut down the endpoint
session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)



## Step 4 (B): Train the Linear model
linear_container = get_image_uri(session.boto_region_name, 'linear-learner')

linear = sagemaker.estimator.Estimator(linear_container,
                                       role,
                                       train_instance_count=1,
                                       train_instance_type='ml.m4.xlarge',
                                       output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                       sagemaker_session=session)

linear_container.set_hyperparameters(feature_dim=13,
                                     predictor_type='regressor',
                                     mini_batch_size=200)

linear.fit({'train': s3_input_train, 'validation': s3_input_validation})


## Step 6 (B): Deploy the trained model
linear_model_name = "boston-update-linear-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

linear_primary_container = {"Image":linear_container, "ModelDataUrl":linear.model_data}

# And lastly we construct the SageMaker model
linear_model_info = session.sagemaker_client.create_model(ModelName=linear_model_name,
                                                          ExecutionRoleArn=role,
                                                          PrimaryContainer=linear_primary_container)


# Create the endpoint configuration
linear_endpoint_config_name = "boston-linear-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
linear_endpoint_config_info = session.sagemaker_client.create_endpoint_config(EndpointConfigName=linear_endpoint_config_name,
                                                                              ProductionVariants=[{
                                                                                  "InstanceType": "ml.m4.xlarge",
                                                                                  "InitialVariantWeight": 1,
                                                                                  "InitialInstanceCount": 1,
                                                                                  "ModelName": linear_model_name,
                                                                                  "VariantName": "Linear-Model"
                                                                              }])

# Deploy the endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

endpoint_info = session.sagemaker_client.create_endpoint(EndpointName=endpoint_name,
                                                         EndpointConfigName=linear_endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)


## Step 7 (B): Use the model
response = session.sagemaker_runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                                            ContentType="text/csv",
                                                            Body=','.join(map(str, X_test.values[0])))

print(response)
result = response['Body'].read().decode('utf-8')
print(result)
print(Y_test.values[0])

# Shut down the endpoint
session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)


## Step 6 (C): Deploy a combined model
combined_endpoint_config_name = "boston-combined-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
combined_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = combined_endpoint_config_name,
                            ProductionVariants = [
                                { # First we include the linear model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": linear_model_name,
                                    "VariantName": "Linear-Model"
                                }, { # And next we include the xgb model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": xgb_model_name,
                                    "VariantName": "XGB-Model"
                                }])

endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = combined_endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)


## Step 7 (C): Use the model
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[0])))
pprint(response)

for rec in range(10):
    response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[rec])))
    pprint(response)
    result = response['Body'].read().decode("utf-8")
    print(result)
    print(Y_test.values[rec])


# To figure out the properties of the deployed endpoint
print(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))


# Updating an Endpoint
# Let's say we decided Linear model is working better than XGBoost as a result of A/B test. Now, we want to switch from combined model
# to Linear model only in our endpoint. Instead of shutting down the current endpoint and starting linear one, we update th endpoint.
session.sagemaker_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=linear_endpoint_config_name)

print(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))

endpoint_dec = session.wait_for_endpoint(endpoint_name)

print(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))

# Shut down the endpoint
session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)


## Optional: Clean up
# First we will remove all of the files contained in the data_dir directory
#!rm $data_dir/*

# And then we delete the directory itself
#!rmdir $data_dir