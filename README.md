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

##`datasets` contains:
1. `array_generator` which takes `.hdf5` files from edep-sim and converts into files with max 500 arrays of shape (3, 64, 64). Also contains `vis.py`, to plot and save images as examples
2. `training` which starts empty but can hold training data 
3. `testing` which starts empty but can hold testing data

##`main` contains:
1. `train` which has submit scripts for training models
To train a model, make a new directory inside `train` and copy the appropriate submit script inside. Edit this submit script to point to your home directory (substitute `whoami` for your username). If necessary, edit the `train/pytorch_wrapper.sh` to include any arguments. Optionally, add a model into `model_options.py` and be sure to add a keyword into `model_dict` inside `model_options.py` to call the correct model in the argument.

The following options are available in `train_nn.py` (type `python3 train_nn.py -h` to see options.). 
> Train a convolutional neural network for neutrino interactions.
> options:
  -h, --help            show this help message and exit
  --seed SEED           Set random seed (default: 5)
  --numu_folder [NUMU_FOLDER ...]
                        Name of folder containing Nu_mu interactions (default: /home/sthoma31/neutrino_interaction_images/array_generator/numu_10_22)
  --nue_folder [NUE_FOLDER ...]
                        Name of folder containing Nue interactions (default: /home/sthoma31/neutrino_interaction_images/array_generator/nue_10_22)
  --batch_size BATCH_SIZE
                        Batch size when training (default: 128)
  --epoch EPOCH         Number of epochs when training (default: 300)
  --testing_data_size TESTING_DATA_SIZE
                        Size of testing data for each channel (default: 1000)
  --training_size TRAINING_SIZE
                        Approx. size of training dataset (default: 1000)
  --png_header PNG_HEADER
                        Header name for PNG files (default: trial)
  --plot_freq PLOT_FREQ
                        Plot confusion matrices every {plot_freq} times (default: 5)
  --model_name MODEL_NAME
                        Name of model to save as (default: run_number_1)
  --model_file MODEL_FILE
                        Model to use to train (default: modeldw)
  --move_files MOVE_FILES
                        Sort into testing and training files (default: False)
  --testing_directory TESTING_DIRECTORY
                        Testing directory (default: /home/sthoma31/neutrino_interaction_CNN/datasets/testing)
  --training_directory TRAINING_DIRECTORY
                        Training directory (default: /home/sthoma31/neutrino_interaction_CNN/datasets/training)
  --random_order RANDOM_ORDER
                        Randomize order of training data (default: True)
  --device DEVICE       device to run file on (default: GPU)

2. `test` which contains code to produce confusion matrices.

The following options are available in `confusionmatrices.py`.

> Generate confusion matrices

> options:
  -h, --help            show this help message and exit
  --testing TESTING     Testing directory (default: /home/sthoma31/neutrino_interaction_CNN/datasets/testing)
  --training TRAINING   Training directory (default: /home/sthoma31/neutrino_interaction_CNN/datasets/training)
  --model_directory MODEL_DIRECTORY
                        Path to directory with models (default: /home/sthoma31/neutrino_interaction_CNN/main/train)
  --device DEVICE       device to run file on (default: CPU)

`singularity_files` contains: 
1. `singularity_wrapper.sh` which will run your script in a singularity shell
2. place your singularity container here as well if desired. 