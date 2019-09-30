# game-analyser
Birkbeck college final year Msc Project - Intelligent Game Analyser

# packages to be installed for running Mask R-CNN
https://github.com/nramak01/game-analyser/blob/master/requirements.txt

>> python3 setup.py install

# Custom Badminton Dataset
Training Dataset location: https://github.com/nramak01/game-analyser/tree/master/samples/dataset/train
Validation Dataset location: https://github.com/nramak01/game-analyser/tree/master/samples/dataset/val

# Source code location

Source code location for Analyser project : https://github.com/nramak01/game-analyser/tree/master/samples/analyser
Customised class: analyser.py

# Command to Train the model
# Please kindly note due to size constraints 
# unable to upload pre trained coco weight(mask_rcnn_coco.h5) and trained analyser (mask_rcnn_analyser.h5)
>> python3 analyser.py train --dataset=/samples/dataset --weights=coco
# Test Images
Below location contains all the images used for testing
Test Image location: https://github.com/nramak01/game-analyser/tree/master/Testing_Results
