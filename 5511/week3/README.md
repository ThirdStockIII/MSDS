# In case you don't feel like scrolling through the notebook to find the responses

# Histopathologic Cancer Detection using CNNs



## Problem

I must create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The data for this competition is a slightly modified version of the PatchCamelyon (PCam) benchmark dataset (the original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates)

## Data

In this dataset, I was provided with a large number of small pathology images to classify. Files are named with an image id. The train_labels.csv file provides the ground truth for the images in the train folder. I am predicting the labels for the images in the test folder. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

* Train Size: 220025
* Test Size: 57458

## Exploratory Data Analysis (EDA) — Inspect, Visualize and Clean the Data

This is a very popular dataset as this challenge is one of the more popular ones for people to start using CNNs. Because of the popularity, there isn't anything that needs to be done to clean the data. There aren't any missing or null values, everything is prepped to begin working.

There is some to do to understand the data. With the help of visualizations, we can understand through a pie chart the number of Cancer vs Non-Cancer that exist in the train data set. Furthermore, just being able to see some of the sample images helped me understand what exactly I am working with to visualize cancer so I can adjust the model to help it do the same.

## Model Architecture

I'm using a Deep Convolutional Neural Network for this task building which is fairly straight-forward in PyTorch if you understand how it works. This is one of many architectures I tried that gave better results.

## Results and Analysis

The AUC score I was able to achieve with this model on test set is ~0.95 which shows the model is actually predicting with accuracy instead of just randomly guessing correctly. This project has given me confidence that I can continue to work and improve, but the model also took over 21 hours to run, so I will have to revisit this at a much later time to allow my computer to recover.

## Conclusion

I had a lot of fun working through this project. I am very new to Machine Learning and Deep Learning was something I never thought I would be able to produce anything with. Going through guides and reading how other people have solved this beginner problem was great for me to be able to feel inspired and to learn myself. 

Aside from how long it took to actually run, which might mean I need a better GPU, but honestly, not in this economy. I would like to continue with another CNN problem to see what else I could learn. 
