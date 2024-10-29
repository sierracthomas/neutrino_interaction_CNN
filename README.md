# A repository for training and testing CNN models on neutrino interactions

To get this repository, navigate to your home directory and type 

`git clone git@github.com:sierracthomas/neutrino_interaction_CNN.git` 

Be sure to have an available conda environment called `pytorch`. Instructions for this are [located in the Syracuse Orange Grid Examples repository.](https://github.com/SyracuseUniversity/OrangeGridExamples/tree/main/PyTorch)

Inside the conda environment (via `conda activate pytorch`), install some python libraries. 
`conda install matplotlib argparse time seaborn pandas`

Note that available testing data and training data (for 700 MeV events) are located in the following repositories: 

- `/home/sthoma31/neutrino_interaction_images/array_generator/numu_10_7`
- `/home/sthoma31/neutrino_interaction_images/array_generator/numu_10_22`
- `/home/sthoma31/neutrino_interaction_images/array_generator/nue_10_7` 
- `/home/sthoma31/neutrino_interaction_images/array_generator/nue_10_22`


# Table of contents: 

## `datasets` contains:
1. `array_generator` which takes `.hdf5` files from edep-sim and converts into files with max 500 arrays of shape (3, 64, 64). Also contains `vis.py`, to plot and save images as examples
2. `training` which starts empty but can hold training data 
3. `testing` which starts empty but can hold testing data

## `main` contains:
1. `train` which has submit scripts for training models
To train a model, make a new directory inside `train` and copy the appropriate submit script inside. Edit this submit script to point to your home directory (substitute `whoami` for your username). If necessary, edit the `train/pytorch_wrapper.sh` to include any arguments. Optionally, add a model into `model_options.py` and be sure to add a keyword into `model_dict` inside `model_options.py` to call the correct model in the argument.

Type `python3 train_nn.py -h` to see options.

2. `test` which contains code to produce confusion matrices.

Type `python3 confusionmatrices.py -h` to see options.

`singularity_files` contains: 
1. `singularity_wrapper.sh` which will run your script in a singularity shell
2. place your singularity container here as well if desired. 