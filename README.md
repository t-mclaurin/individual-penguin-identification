crop.py - copys and produces crops from images using megadetecor bounding boxes. 
database_preprocess.py - copies and applies histogram normalisation to a directory of images, retaining the same organisational structure. 
medoid_poles_vis_embeddings.py - embedds, labels key points and visualises a folder of images, returns file paths of key photos. 
model.py - defines the build_embedding_model function.
pop_est_test.py - Visualises the population estimate and error rates across a range of threshold values for a known set of images. 
unlabelled_pop_est_test.py - Visualses the popualtion estimates across a range of threshold values for an unknwon set of images. 
tag_occluder.py - GUI for the manual application of black squares to occlude tags from images, produces a copy of the input directory with the same structure. Can be saved. 
train_v4_gridsearch.py - Performs a gridsearch across hyperparamters and a model on each varation to an adaptive stopping point. Requires a csv of dataset split metadata and training data. Returns model weights and a csv of training and validation metrics.  
updated_visualise_embeddings.py - monocolour t-snes visualisation of embeddings, from a directoy of images. 
colour_visualise_embeddings.py - t-snes visualisation of embeddings, coloured acroroding to dataset split metadata, from a directory of images. 
validate_crops.py - Copies and sorts images into a vlaid and invalid subdirectories according to the judegment of a loaded binary image classfier and the set threshold. 

Utils:
create_tf_data_csv.py - creates a csv file of train, test and validation metadata labels for a directory of images. 
mining.py - provides various functions for generating triplets. 
augmentation.py - provides various image augmentation functions. 
