#%matplotlib inline
import os
import time
from time import gmtime, strftime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

## Step 3: Uploading the data files to S3
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

X_test.to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

prefix = 'boston-xgboost-tuning-HL'
test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)


## Step 4: Train and construct the XGBoost model
container = get_image_uri(session.boto_region_name, 'xgboost')
training_params = {
    "RoleArn": role,
    "AlgorithmSpecification": {"TrainingImage": container, "TrainingInputFile": "File"},
    "OutputDataConfig": {"S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"},
    "ResourceConfig": {"InstanceCount": "1", "InstanceType": "ml.m4.xlarge", "VolumeSizeInGB": "5"},
    "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
    "StaticHyperParameters": {"gamma": "4", "subsample": "0.8", "objective": "reg:linear",
                              "early_stopping_rounds": "10", "num_round": "10"},
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": train_location, "S3DataDistributionType": "FullyReplicated"}
            },
            "ContentType": "csv",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": val_location,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "csv",
            "CompressionType": "None"
        }
    ]
}

# We need to construct a dictionary which specifies the tuning job we want SageMaker to perform
tuning_job_config = {
    # First we specify which hyperparameters we want SageMaker to be able to vary,
    # and we specify the type and range of the hyperparameters.
    "ParameterRanges": {
    "CategoricalParameterRanges": [],
    "ContinuousParameterRanges": [
        {
            "MaxValue": "0.5",
            "MinValue": "0.05",
            "Name": "eta"
        },
    ],
    "IntegerParameterRanges": [
        {
            "MaxValue": "12",
            "MinValue": "3",
            "Name": "max_depth"
        },
        {
            "MaxValue": "8",
            "MinValue": "2",
            "Name": "min_child_weight"
        }
    ]},
    # We also need to specify how many models should be fit and how many can be fit in parallel
    "ResourceLimits": {
        "MaxNumberOfTrainingJobs": 20,
        "MaxParallelTrainingJobs": 3
    },
    # Here we specify how SageMaker should update the hyperparameters as new models are fit
    "Strategy": "Bayesian",
    # And lastly we need to specify how we'd like to determine which models are better or worse
    "HyperParameterTuningJobObjective": {
        "MetricName": "validation:rmse",
        "Type": "Minimize"
    }
  }

# Execute the tuning job
tuning_job_name = "tuning-job" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
session.sagemaker_client.create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name,
                                                           HyperParameterTuningJobConfig = tuning_job_config,
                                                           TrainingJobDefinition = training_params)

session.wait_for_tuning_job(tuning_job_name)

# Build the model
tuning_job_info = session.sagemaker_client.describe_hyper_parameter_tunuing_job(HyperParameterTuningJobName=tuning_job_name)
best_training_job_name = tuning_job_info["BestTrainingJob"]["TrainingJobName"]

model_artifacts = tuning_job_info["ModelArtifacts"]["S3ModelArtifacts"]

model_name = best_training_job_name + "-model"

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the xgboost container that we used for training
# can also be used for inference
primary_container = {"Image": container, "ModelDataUrl": model_artifacts}

# And lastly we construct the SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)


## Step 5: Testing the model
# Just like in each of the previous steps, we need to make sure to name our job and the name should be unique.
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Now we construct the data structure which will describe the batch transform job.
transform_request = \
    {
        "TransformJobName": transform_job_name,

        # This is the name of the model that we created earlier.
        "ModelName": model_name,

        # This describes how many compute instances should be used at once. If you happen to be doing a very large
        # batch transform job it may be worth running multiple compute instances at once.
        "MaxConcurrentTransforms": 1,

        # This says how big each individual request sent to the model should be, at most. One of the things that
        # SageMaker does in the background is to split our data up into chunks so that each chunks stays under
        # this size limit.
        "MaxPayloadInMB": 6,

        # Sometimes we may want to send only a single sample to our endpoint at a time, however in this case each of
        # the chunks that we send should contain multiple samples of our input data.
        "BatchStrategy": "MultiRecord",

        # This next object describes where the output data should be stored. Some of the more advanced options which
        # we don't cover here also describe how SageMaker should collect output from various batches.
        "TransformOutput": {
            "S3OutputPath": "s3://{}/{}/batch-bransform/".format(session.default_bucket(), prefix)
        },

        # Here we describe our input data. Of course, we need to tell SageMaker where on S3 our input data is stored, in
        # addition we need to detail the characteristics of our input data. In particular, since SageMaker may need to
        # split our data up into chunks, it needs to know how the individual samples in our data file appear. In our
        # case each line is its own sample and so we set the split type to 'line'. We also need to tell SageMaker what
        # type of data is being sent, in this case csv, so that it can properly serialize the data.
        "TransformInput": {
            "ContentType": "text/csv",
            "SplitType": "Line",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": test_location,
                }
            }
        },

        # And lastly we tell SageMaker what sort of compute instance we would like it to use.
        "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
        }
    }

transform_response = session.sagemaker_client.create_transform_job(**transform_request)
transform_desc = session.wait_for_transform_job(transform_job_name)

# Analyze the results
transform_output = "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
#!aws s3 cp --recursive $transform_output $data_dir

Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")


## Optional: Clean up
# First we will remove all of the files contained in the data_dir directory
#!rm $data_dir/*

# And then we delete the directory itself
#!rmdir $data_dir