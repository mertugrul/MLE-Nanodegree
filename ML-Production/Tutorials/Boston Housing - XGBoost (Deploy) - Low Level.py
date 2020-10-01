#%matplotlib inline

import os
import time
from time import gmtime, strftime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import sklearn.model_selection

## SageMaker session initialization
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# This is an object that represents the SageMaker session that we are currently operating in. This
# object contains some useful information that we will need to access later such as our region.
session = sagemaker.Session()

# This is an object that represents the IAM role that we are currently assigned. When we construct
# and launch the training job later we will need to tell it what IAM role it should have. Since our
# use case is relatively simple we will simply assign the training job the role we currently have.
role = get_execution_role()

## Download the data, Split the data, Save the data as csv.
boston = load_boston()

X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

# We split the dataset into 2/3 training and 1/3 testing sets.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_bos_pd, Y_bos_pd, test_size=0.33)

# Then we split the training set further into 2/3 training and 1/3 validation sets.
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.33)

data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)

## Upload the data to S3
prefix = 'boston-xgboost-deploy-ll'
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
validation_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)

# We will need to know the name of the container that we want to use for training. SageMaker provides
# a nice utility method to construct this for us.
training_job_name = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
container = get_image_uri(session.boto_region_name, 'xgboost')

training_params = {
    "TrainingJobName": training_job_name,
    "RoleArn": role,
    "AlgorithmSpecification": {"TrainingImage": container, "TrainingInputMode": "File"},
    "OutputDataConfig": {"S3OutputPath": "s3://{}/{}/output".format(session.default_bucket(), prefix)},
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": train_location,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "csv",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource":{
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": validation_location,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "csv",
            "CompressionType": "None"
        }
    ],
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.m4.xlarge",
        "VolumeSizeInGB": 5
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
    "HyperParameters": {
        "max_depth": "5",
        "eta": "2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.8",
        "objective": "reg:linear",
        "early_stopping_rounds": "10",
        "num_round": "200"
    }
}

## Execute the Training Job
training_job = session.sagemaker_client.create_training_job(**training_params)
# if we want to wait until it finishes:
session.logs_for_job(training_job_name, wait=True)

## Build the model
# We begin by asking SageMaker to describe for us the results of the training job. The data structure
# returned contains a lot more information than we currently need, try checking it out yourself in
# more detail.
training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']

# Just like when we created a training job, the model name must be unique
model_name = training_job_name + "-model"

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the xgboost container that we used for training
# can also be used for inference.
primary_container = {
    "Image": container,
    "ModelDataUrl": model_artifacts
}

# And lastly we construct the SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)


## Create and deploy the endpoint
# First create and endpoint configuration with a unique name. Then, the same endpoint conf. can be used as a blueprint with different endpoints.
endpoint_config_name = 'boston-xgboost-endpoint-config-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
endpoint_config_info = session.sagemaker_client.create_endpoint_config(EndpointConfigName=endpoint_config_name,
                                                                       ProductionVariants=[{"InstanceType": "ml.m4.xlarge",
                                                                                            "InitialVariantWeight": 1,
                                                                                            "InitialInstanceCount": 1,
                                                                                            "ModelName": model_name,
                                                                                            "VariantName": "AllTraffic"}])

endpoint_name = 'boston-xgboost-endpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
endpoint_info = session.sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
endpoint_dec = session.wait_for_endpoint(endpoint_name)

## Use the model
# Serialize the test data
payload = [[str(entry) for entry in row] for row in X_test.values]
payload = '\n'.join([','.join(row) for row in payload])

response = session.sagemaker_runtime_client.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload)

result = response['Body'].read().decode("utf-8")
y_pred = np.fromstring(result, sep=',')

plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")

## Delete the endpoint
session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

## Optional: Clean up
# First we will remove all of the files contained in the data_dir directory
#!rm $data_dir/*

# And then we delete the directory itself
#!rmdir $data_dir
