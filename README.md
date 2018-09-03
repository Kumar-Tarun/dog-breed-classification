This is my script for dog breed identification competition on Kaggle.

Directory Structure:

main.py : Contains the model, training and testing part.<br/>
train_generator.py : Contains generator for training images.<br/>
test_generator.py : Contains generator for testing images.<br/>

lables.csv : Contains the label for training images (Downloaded from Kaggle)<br/>
classes.npy : Contains all the dog classes.<br/>
test_ids.npy : Contains the ids of test images.<br/>

train/ : Should contain the training images.<br/>
test/ : Should contain the testing images.<br/>

Pre-trained model used: Xception<br/>
Multi-Class Log Loss: 0.46222<br/>
Training Accuracy: ~90% (on 9500 images) <br/>
Validation Accuracy: ~86% (on 700 images) <br/>


