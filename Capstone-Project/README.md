# Dog-breed-classification using CNN and transfer learning
This is the repo of dog breed classifier for Capstone project of Udaicty's Machine Learning Engineer nanodegree program.

## Project Overiview
Dog breed identification or classification is a typical task for object detection/recognition. The purpose of this project is to classify the dog breeds based on a given picture of a dog as the input. Such an application can be used within web app to process real-world, user-supplied images. It would be very useful for a large audience (e.g., children) who want to learn and investigate more about the dogs. This project aims to achieve two main tasks as follows:

* To estimate the canine’s breed based on the given image of a dog with a reasonable accuracy.
* To estimate the most resembling canine’s breed if the supplied image (input) is an image of a human.

To solve this problem: (1) A human-face detector is developed using one of the Haar feature-based cascade classifiers of the OpenCV. (2) A dog-detector is implemented to identify if there exists a dog in the given input image. For this task, a pretrained VGG16 model used. (3) Two separate CNN models are developed to estimate the dog breed for the given input image. First, a CNN-based dog breed classifier with three convolutional layers and two fully connected layers are trained from scratch. Second, another dog breed classifier is implemented using transfer learning where a pretrained VGG16 models is finetuned on the dog-image dataset. 

## Dataset
Two dataset are used in this project: (1) Dog-image dataset, (2) human-image dataset. These datasets can be accessed from the links below:
* Dog-image dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
* Human-image dataset: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip

Dog-image dataset includes 8351 dog images for 133 different breeds. The dataset is divided into three sets, training, validation and test, where there are 6680, 835, and 836 images for training, validation and test sets, respectively. Human-image dataset includes 13233 human images from 5749 different people.

## Model 1: Dog breed classifier from scratch
A CNN is implemented including three convolutional layers and followed by two fully connected layers. Each convolutional layer consists of a convolution operation, a 2d batch normalization operation, a ReLU operation, a max pooling operation and a dropout with 0.2. In the first convolutional operation, `kenerl_size` and `stride` are set to 5 and 2, respectively. For other layers, `kenerl_size` and `stride` are set to 5 and 1, respectively. For each max pooling operation, `kenerl_size` and `stride` are set to 2. Through the convolutional layers, the number of channels/kernels increases (i.e., 3, 64, 128, 128) based on the best practices.

## Model 2: Dog breed classifier using transfer learning
Aa pretrained VGG16 model is used as the base, and it is fintuned on the dog-image training dataset. Only the last fully connected layer is changed. The parameters of all layers are frozen (except for the new added last fully connected layer). The input size and output size of the last fully connected layer are 4096 and 133, respectively.

## Evaluation
The Model 1 (Dog breed classifier from scratch) yielded 22% accuracy on the test set. On the other hand, Model 2 (Dog breed classifier using transfer learning) performed 82% accuracy on the test set. Two example outputs of the Model 2 are shown below:

![alternativetext](images/dog1.png)
![alternativetext](images/human1.png)


## Requirements
Minimum required versions of the respective tools are: 
* `numpy >= 1.12.1`
* `cv2 >= 3.3.1`
* `matplotlib >= 2.1.0`
* `tqdm >= 4.11.2`
* `torch >= 0.4.0`
* `torchvision >= 0.2.1`
* `PIL >= 5.2.0`

Also, include `glob` for file operations.
