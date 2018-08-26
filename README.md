This is my script for dog breed identification competition on Kaggle.

Directory Structure:

main.py : Contains the model, training and testing part.
train_generator.py : Contains generator for training images.
test_generator.py : Contains generator for testing images.

lables.csv : Contains the label for training images (Downloaded from Kaggle)
classes.npy : Contains all the dog classes.
test_ids.npy : Contains the ids of test images.

train/ : Should contain the training images.
test/ : Should contain the testing images.

Pre-trained model used: Xception
Multi-Class Log Loss: 0.46222


