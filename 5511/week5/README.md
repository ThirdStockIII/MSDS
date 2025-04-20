# In case you don't feel like scrolling through the notebook to find the responses

# I’m Something of a Painter Myself

We recognize the works of artists through their unique style, such as color choices or brush strokes. The “je ne sais quoi” of artists like Claude Monet can now be imitated with algorithms thanks to generative adversarial networks (GANs).

Computer vision has advanced tremendously in recent years and GANs are now capable of mimicking objects in a very convincing way. But creating museum-worthy masterpieces is thought of to be, well, more art than science. So can (data) science, in the form of GANs, trick classifiers into believing you’ve created a true Monet?

## Problem

A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For our competition, you should generate images in the style of Monet. This generator is trained using a discriminator.

The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.

The task is to build a GAN that generates 7,000 to 10,000 Monet-style images.

## Data

The dataset contains four directories: monet_tfrec, photo_tfrec, monet_jpg, and photo_jpg. The monet_tfrec and monet_jpg directories contain the same painting images, and the photo_tfrec and photo_jpg directories contain the same photos.

The monet directories contain Monet paintings. These images are going to be used to train my model.

The photo directories contain photos. I will add Monet-style to these images to test my model.

To explain the datasets more specifically:

* monet_jpg - 300 Monet paintings sized 256x256 in JPEG format
* monet_tfrec - 300 Monet paintings sized 256x256 in TFRecord format
* photo_jpg - 7028 photos sized 256x256 in JPEG format
* photo_tfrec - 7028 photos sized 256x256 in TFRecord format

## Exploratory Data Analysis (EDA) — Inspect, Visualize and Clean the Data

I don't think there was much necessity to clean any of the data. It was just images after all that were all the same size

I was able to verify that size and then compare a regular photo with a Monet to help understand the differences between the two before I begin to build a model that adds dimensions to the regular photos to resemble the artist's work. 

## Model Architecture

The goal is to create a Generator and a Discriminator. The Generators purpose is to add a lot of static and colors so that there are more layers than the natural picture has. Using a U-Net allows for image segmentation so that there I can reconstruct the spatial information.

After ensuring that the model is successful in this stage, I work on building a discriminator so that the image actually looks like a Monet. The goal is to tune the hyperparamters until I find a model that works the best. And boy did this take forever to run. Like over a day. I can't be bothered to change anything and even typing this feels exhausting because I just want to submit it.

## Results and Analysis

The final Epoch game me the results of:
Epoch: 20 | Generator Loss: 2.26104163368543 | Discriminator Loss: 0.4417468906597545
compared to the first generation of:
Epoch: 1 | Generator Loss: 3.2388192137082417 | Discriminator Loss: 0.7003519776463508

Overall the model was able to improve significantly. I was very much asleep during the time the model was compiling, but there is definitely evidence that the model was able to build and improve over the renditions. 

## Conclusion

Overall this project was interesting. I really wanted to work on the one with Dogs, but I am glad that I took the time to actually read the assignment where it clearly said that the dataset on Kaggle no longer allows for submissions to check the work. 

Working with images was very unique. It was weird not having to do any real data cleaning. I think in the future it would be fun to work with a "dataset" where the pictures aren't all the same size. That might allow for some work looking for outliers that would take too long to generate a new image of or maybe to check what images aren't similar to the ones you are trying to work with. Fun project, felt accomplished that I was able to make something and learn more about deep learning.
