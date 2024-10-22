# A repository for training and testing CNN models on neutrino interactions

# Table of contents: 
`datasets` contains:
1. `array_generator` which takes `.hdf5` files from edep-sim and converts into files with max 500 arrays of shape (3, 64, 64). Also contains `vis.py`, to plot and save images as examples
2. `training` which starts empty but can hold training data 
3. `testing` which starts empty but can hold testing data

`main` contains:
1. `train` which has submit scripts for training models
2. `test` which contains code to produce confusion matrices 