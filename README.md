# Banded Penguin Identification Project 

This Document summarises the workflow and scripts of the python project behind the research and application of my (Theo McLaurin's) MRes masters research project.

The Project aimed to build and train a computer vision embedding model that could
identify individual Banded Penguins (Spheniscus) from manual photographs and camera traps.
I aimed to build a model that could work with varation across age, pose and image quality.
Testing shows that the final model weights were capable of solving the open-set 
recognition problem commonly dealt with by conservation technologists. 

## How to recreate the results of the Project
This is a step by step process to recreate the process undertaken by me to build and train a covolutional neural network to identify individual Banded Penguins from photos. A large source of inspiration was the work done by Olga Moskvyak (https://github.com/olgamoskvyak) on Manta Rays. This approah is generalisable to any and all open-set re-identification tasks, not just the species studied by Olga and I. For the rest of this tutorial I will assume you are working with a speices of Banded penguin. 

**Pre-requesits**
You will need a database of Known Individuals, and photos of them across a wide variety of poses and environments. There is no rule about how many identities and how many photos is necessary, however the more the bettter. I utilised easy access to a known popualtion of Humboldt penguins at ZSL London Zoo to build a dataabse of 2219 images of 81 individuals. This database should exist as a directory of IDs, each ID contianing the images fo that individual.

ENV file? 

**Occluding Tags**
My Zoo housed subjects had identifying wing bands, for which I build a duplicate second dataset that had these tags covered by a black square in order to focus the model on the features of the penguin and help it transfer to applications on wild popualtions. 

tag_occluder.py - Runs a GUI that displays one image at time from the source directory. Shift and Option keys toggle the size of the occluding square. Each photo is duplicated and stored in the destination directory in the same structure. The Progress and metadata is saved in a .csv file so that the (often tedious) process is saved, the script can be quit and returned to and it will resume progress. 

**Histogram Normalisation**
Pre-processing the images with Histogram Normalisation often makes the image dataset reduces the variability of image characteristics and increases the contrast. This has often been found to assist computer vision models, however I found that it had no effect on model perfomance for my identification models. 

database_preprocess - Duplicates the content and structure of a database and applies histogram normalsiation to all images.

**Split the Dataset**

We need to deliniate photos that are to be used to train the model and to test the mdoel. I perfrom two tests using photos that the model hasn't seen before, one on individuals it has seen beofre and one on individuals it hasn't seen before. As such we need to first divide the individuals into "known" and "unknown" and then the "known" individuals into test and train. I set the identities with the fewest images to be the unknown set, which may not be suitable for your dataset. 

create_tf_data_csv.py -- Reads the image dataset and creates an unknown/known based on the number fo images of the identity and then test/train dataset split written to a csv file. The file contains pathnames and the affomentioned labels. 

**Perform a model training hyper-paramater grid-search**

We can now train our models. The hyper-paramter grid search accepts a large number of paramter variables, but takes a simple approach of training each combination to completion. The variables that can be explored are: 

| Parameter      | Function                                  |
|----------------|-------------------------------------------|
| LEARNING_RATE  | Head-only Learning rate                   |
| LEARNING_RATE_2| Head and Backbone Learning rate           |
| MARGIN         | Triplet Loss Margin                       |
| WARMUP_LENGTH  | Epochs run before unfreezing the Backbone |
| DROPOUT_RATE   | Dropout Rate for the Head                 |



![IMG_1953_box2](https://github.com/user-attachments/assets/8457a3e2-0264-4859-9d71-6a8dd43b2956)

Glossary
Main Scripts

crop.py
Copies and produces crops from images using MegaDetector bounding boxes.

database_preprocess.py
Copies and applies histogram normalisation to a directory of images, retaining the same organisational structure.

medoid_poles_vis_embeddings.py
Embeds, labels key points, and visualises a folder of images. Returns file paths of key photos.

model.py
Defines the build_embedding_model function.

pop_est_test.py
Visualises the population estimate and error rates across a range of threshold values for a known set of images.

unlabelled_pop_est_test.py
Visualises the population estimates across a range of threshold values for an unknown set of images.

tag_occluder.py
GUI for the manual application of black squares to occlude tags from images. Produces a copy of the input directory with the same structure. Can be saved.

train_v4_gridsearch.py
Performs a grid search across hyperparameters and a model on each variation to an adaptive stopping point.
Requires a CSV of dataset split metadata and training data.
Returns model weights and a CSV of training and validation metrics.

updated_visualise_embeddings.py
Monochrome t-SNE visualisation of embeddings, from a directory of images.

colour_visualise_embeddings.py
t-SNE visualisation of embeddings, coloured according to dataset split metadata, from a directory of images.

validate_crops.py
Copies and sorts images into valid and invalid subdirectories according to the judgement of a loaded binary image classifier and the set threshold.

Utils

create_tf_data_csv.py
Creates a CSV file of train, test, and validation metadata labels for a directory of images.

mining.py
Provides various functions for generating triplets.

augmentation.py
Provides various image augmentation functions.
