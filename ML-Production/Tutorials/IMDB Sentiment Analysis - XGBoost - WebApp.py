import os
import glob
#from sklearn.externals import joblib
import joblib
from sklearn.utils import shuffle
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import sagemaker
from sklearn.metrics import accuracy_score


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

## Prepare and Processing the data
def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)

            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                "{}/{} data size does not match labels size".format(data_type, sentiment)

    return data, labels

data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""

    # Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']

    # Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)

    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test

train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
print(train_X[100])

def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words

print(review_to_words(train_X[100]))

cache_dir = os.path.join("../cache", "sentiment_web_app")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists


def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        # words_train = list(map(review_to_words, data_train))
        # words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]

        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                                                              cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test

# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)


## Extract Bag-of-Words features
def extract_BoW_features(words_train, words_test, vocabulary_size=5000,
                         cache_dir=cache_dir, cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(max_features=vocabulary_size)
        features_train = vectorizer.fit_transform(words_train).toarray()

        # Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test = vectorizer.transform(words_test).toarray()

        # NOTE: Remember to convert the features using .toarray() for a compact representation

        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                              vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                                                     cache_data['features_test'], cache_data['vocabulary'])

    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary

# Extract Bag of Words features for both training and test datasets
train_X, test_X, vocabulary = extract_BoW_features(train_X, test_X)
print(len(train_X[100]))


## Step 3: Upload the data to S#
# Earlier we shuffled the training dataset so to make things simple we can just assign
# the first 10 000 reviews to the validation set and use the remaining reviews for training.
val_X = pd.DataFrame(train_X[:10000])
train_X = pd.DataFrame(train_X[10000:])

val_y = pd.DataFrame(train_y[:10000])
train_y = pd.DataFrame(train_y[10000:])

data_dir = "../data/sentiment_web_app"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

pd.DataFrame(test_X).to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)
pd.concat([val_y, val_X], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([train_y, train_X], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

# To save a bit of memory we can set text_X, train_X, val_X, train_y and val_y to None.
test_X = train_X = val_X = train_y = val_y = None

## Uploading Training / Validation files to S3
session = sagemaker.Session()
prefix = 'sentiment-web-app'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)

## Step 4: Creating the XGBoost model
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
role = get_execution_role()
container = get_image_uri(session.boto_region_name, 'xgboost')

xgb = sagemaker.estimator.Estimator(container,
                                    role,
                                    train_instance_count=1,
                                    train_instance_type='ml.m4.xlarge',
                                    output_path="S3://{}/{}/output".format(session.default_bucket(), prefix),
                                    sagemaker_session=session)

xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_depth=6,
                        subsample=0.8,
                        silent=0,
                        objective="binary:logistic",
                        early_stopping_rounds=10,
                        num_round=500)

# Fit the XGBoost model
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({'train':s3_input_train, 'validation': s3_input_validation})


## Step 5: Testing the model
xgb_transformer = xgb.transformer(instance_count=1, instance_type='ml.m4.xlarge')
xgb_transformer.transform(test_location, content_type='csv', split_type="Line")
xgb_transformer.wait()

# Get output to the dictionary
#!aws s3 cp --recursive $xgb_transformer.output_path $data_dir

predictions = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]

accuracy_score(test_y, predictions)


## Step 6: Deploying the model
xgb_predictor = xgb.deploy(initial_instance_count=1, initial_instance_type='ml.m4.xlarge')

# Testing the model again
from sagemaker.predictor import csv_serializer
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer


# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:], sep=',')

test_X = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None).values

predictions = predict(test_X)
predictions = [round(num) for num in predictions]

accuracy_score(test_y, predictions)

## Cleaning up
xgb.predictor.delete_endpoint()



#### Step 7: Putting our model to work

test_review = "Nothing but a disgusting materialistic pageant of glistening abed remote control greed zombies, totally devoid of any heart or heat. A romantic comedy that has zero romantic chemestry and zero laughs!"
test_words = review_to_words(test_review)
print(test_words)

def bow_encoding(words, vocabulary):
    bow = [0] * len(vocabulary) # Start by setting the count for each word in the vocabulary to zero.
    for word in words.split():  # For each word in the string
        if word in vocabulary:  # If the word is one that occurs in the vocabulary, increase its count.
            bow[vocabulary[word]] += 1
    return bow

test_bow = bow_encoding(test_words, vocabulary)
print(test_bow, len(test_bow))

xgb_predictor.deploy(initial_instance_count=1, initial_instance_type='ml.m4.xlarge')

"""At this point we could just do the same thing that we did earlier when we tested our deployed model and send `test_bow` to our endpoint using 
the `xgb_predictor` object. However, when we eventually construct our Lambda function we won't have access to this object, 
so how do we call a SageMaker endpoint? It turns out that Python functions that are used in Lambda have access to another Amazon library 
called `boto3`. This library provides an API for working with Amazon services, including SageMaker. To start with, we need to get a handle to 
the SageMaker runtime."""

import boto3
runtime = boto3.Session().client('sagemaker-runtime')

response = runtime.invoke_endpoint(EndpointName=xgb_predictor.endpoint, ContentType='text/csv', Body=','.join([str(val) for val in test_bow]).encode('utf-8'))
response = response['Body'].read().decode('utf-8')
print(response)





