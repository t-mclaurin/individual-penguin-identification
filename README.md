# Banded Penguin Identification Project 

This Document summarises the workflow and scripts of the python project behind the research and application of my (Theo McLaurin's) MRes masters research project.

The Project aimed to build and train a computer vision embedding model that could
identify individual Banded Penguins (Spheniscus) from manual photographs and camera traps.
I aimed to build a model that could work with varation across age, pose and image quality.
Testing shows that the final model weights were capable of solving the open-set 
recognition problem commonly dealt with by conservation technologists. 


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
