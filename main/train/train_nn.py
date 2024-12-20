
import argparse
# machine learning libraries
#import torch
from torchvision import datasets, transforms

# import torchvision, which is a complementary tool for image importing in pytorch
import torchvision 
# import numpy for math stuff
#import numpy as np
# import matplotlib for nice plotting
import matplotlib.pyplot as plt
# import time so we can see how long training takes!
from time import time
# import seaborn for other nice plotting
import seaborn as sn
# import pandas for data management (particularly for large sets of data)
import pandas as pd
from model_options import *
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import random
import matplotlib



matplotlib.rcParams.update({'font.size': 22})

homedir = os.path.expanduser('~')


parser = argparse.ArgumentParser(description='Train a convolutional neural network for neutrino interactions. ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", type = int, default = 5, help = "Set random seed")
#parser.add_argument("--image_dir", type = str, default = "/home/sthoma31/neutrino_interaction_images/nu_mu_700/data/QES", nargs='*', help = "Location that contains interaction folders CC and NC")
parser.add_argument("--numu_folder", type = str, default = f"/home/sthoma31/neutrino_interaction_images/array_generator/numu_10_22", nargs='*', help = "Name of folder containing Nu_mu interactions")
parser.add_argument("--nue_folder", type = str, default = f"/home/sthoma31/neutrino_interaction_images/array_generator/nue_10_22", nargs='*', help = "Name of folder containing Nue interactions")
parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size when training")
#parser.add_argument("--training_data_size", type = int, default = 4000, help = "Size of training data")
parser.add_argument("--epoch", type = int, default = 300, help = "Number of epochs when training")
parser.add_argument("--testing_data_size", type = int, default = 1000, help = "Size of testing data for each channel ")
parser.add_argument("--training_size", type = int, default = 1000, help = "Approx. size of training dataset")
parser.add_argument("--png_header", type = str, default = "trial", help = "Header name for PNG files")
parser.add_argument("--plot_freq", type = int, default = 5, help = "Plot confusion matrices every {plot_freq} times")
parser.add_argument("--model_name", type = str, default = "run_number_1", help = "Name of model to save as")
parser.add_argument("--model_file", type = str, default = "modeldw", help = "Model to use to train")
parser.add_argument("--move_files", type = bool, default = False, help = "Sort into testing and training files")

parser.add_argument("--testing_directory", type = str, default = f"{homedir}/neutrino_interaction_CNN/datasets/testing", help = "Testing directory")
parser.add_argument("--training_directory", type = str, default = f"{homedir}/neutrino_interaction_CNN/datasets/training", help = "Training directory")
parser.add_argument("--random_order", type = bool, default = True, help = "Randomize order of training data")

parser.add_argument("--device", type = str, default = "GPU", help = "device to run file on")
args = parser.parse_args()

if args.device == "GPU":
    if torch.cuda.is_available():
        print("Using GPU")
        print(torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        print("Using CPU - exiting program...")
        device = torch.device("cpu")
        exit()

elif args.device == "CPU":
    print("Using CPU - exiting program...")
    device = torch.device("cpu")


torch.manual_seed(args.seed)
#IMAGE_LOC = args.image_dir
BATCH_SIZE = args.batch_size
EPOCH_NUMBER = args.epoch
#datasize = args.training_data_size
testing_data_size = args.testing_data_size
png_header = args.png_header
plot_frequency = args.plot_freq


nuecc_file_list = []
nuecc_shape_list = []
nuenc_file_list = []
nuenc_shape_list = []


# get lengths of arrays in advance!
for filename in os.listdir(args.nue_folder):
    f = os.path.join(args.nue_folder, filename)
    # checking if it is a file
    if os.path.isfile(f):
        if f[-len("labels.npy"):] == "labels.npy":
            temp_load = np.load(f, mmap_mode = "r")
            temp_load = len(temp_load)
            
            if "nuecc" in f:
                nuecc_file_list.append(f)
                nuecc_shape_list.append(temp_load)
                #print("nuecc", temp_load)
            elif "nuenc" in f:
                #print("nuenc", temp_load)
                nuenc_file_list.append(f)
                nuenc_shape_list.append(temp_load)

    
numucc_file_list = []
numucc_shape_list = []
numunc_file_list = []
numunc_shape_list = []

# get lengths of arrays in advance!
for filename in os.listdir(args.numu_folder):
    f = os.path.join(args.numu_folder, filename)
    # checking if it is a file
    if os.path.isfile(f):
        if f[-len("labels.npy"):] == "labels.npy":
            temp_load = np.load(f, mmap_mode = "r")
            temp_load = len(temp_load)
            if "numucc" in f:
                numucc_file_list.append(f)
                numucc_shape_list.append(temp_load)
                #print("nuecc", temp_load)
            elif "numunc" in f:
                #print("nuenc", temp_load)
                numunc_file_list.append(f)
                numunc_shape_list.append(temp_load)

testsize_per_channel = 64

# define test data and training data

if args.move_files: 

    def training_testing_data(total_events = args.testing_data_size, interaction_shape_list = numucc_shape_list, interaction_file_list = numucc_file_list):
        total_events = total_events + 1
        elements_in_total_events = 0
        print("Total interactions: ", np.sum(interaction_shape_list))
        for i in interaction_shape_list:
            total_events = total_events - i
            if total_events < 0:
                break
            elif total_events > 0:
                elements_in_total_events = elements_in_total_events + 1
        print("Testing datasize: ", np.sum(interaction_shape_list[:elements_in_total_events]))
        print("Training datasize: ", np.sum(interaction_shape_list[elements_in_total_events:]))
        return interaction_file_list[elements_in_total_events:], interaction_file_list[:elements_in_total_events]


    # pull out testing data first and the rest is the training data

    print("Numucc: ")
    training_numucc, testing_numucc = training_testing_data(interaction_shape_list = numucc_shape_list, interaction_file_list = numucc_file_list)
    print("Numunc: ")
    training_numunc, testing_numunc = training_testing_data(total_events = int(args.testing_data_size/2), interaction_shape_list = numunc_shape_list, interaction_file_list = numunc_file_list)

    print("Nuecc: ")
    training_nuecc, testing_nuecc = training_testing_data(interaction_shape_list = nuecc_shape_list, interaction_file_list = nuecc_file_list)
    print("Nuenc: ")
    training_nuenc, testing_nuenc = training_testing_data(total_events = int(args.testing_data_size/2), interaction_shape_list = nuenc_shape_list, interaction_file_list = nuenc_file_list)

    training_data_files = training_nuenc + training_nuecc + training_numunc + training_numucc
    testing_data_files = testing_numucc + testing_numunc + testing_nuecc + testing_nuenc
    print("Number of files in training data events: ", len(training_data_files))
    print("Number of files in testing data events: ", len(testing_data_files))

    import shutil

    print("Clearing old directories: ")
    print("testing: ", args.testing_directory)
    print("training: ", args.training_directory)
    if os.path.exists(args.testing_directory):
        shutil.rmtree(args.testing_directory)

    if os.path.exists(args.training_directory):
        shutil.rmtree(args.training_directory)

    print("Making new directories: ")
    if not os.path.exists(args.testing_directory):
        os.makedirs(args.testing_directory)

    if not os.path.exists(args.training_directory):
        os.makedirs(args.training_directory)

    print("Copying training data...")
    for i in training_data_files:
        shutil.copy(i, args.training_directory)
        shutil.copy(i[:-len("_labels.npy")] + ".npy", args.training_directory)

    print("Copying testing data...")
    for i in testing_data_files:
        shutil.copy(i, args.testing_directory)
        shutil.copy(i[:-len("_labels.npy")] + ".npy", args.testing_directory)



elif not args.move_files:
    testing_data_files = []
    training_data_files = []
    # get lengths of arrays in advance!
    for filename in os.listdir(args.testing_directory):
        f = os.path.join(args.testing_directory, filename)
        if os.path.isfile(f):
            if f[-len("labels.npy"):] == "labels.npy":
                testing_data_files.append(f)
    for filename in os.listdir(args.training_directory):
        f = os.path.join(args.training_directory, filename)
        if os.path.isfile(f):
            if f[-len("labels.npy"):] == "labels.npy":
                training_data_files.append(f)

    print(testing_data_files)

### get dataloaders with input data


class CustomDataset(Dataset):
    def __init__(self, img_paths, img_labels, size_of_images):
        
        #self.img_labels = np.load(img_labels)
        self.img_paths = img_paths
        self.size_of_images = size_of_images  
        self.opened_labels = img_labels
        #self.opened_labels = torch.Tensor(self.opened_labels)
        #self.size_of_images = size_of_images
        #if len(self.input_data) != len(self.img_labels):
        #    raise InvalidDatasetException(self.img_paths,self.img_labels)
            
    def __len__(self):
        return len(self.opened_labels)
    def __getitem__(self, index):
        #self.opened_data = np.load(self.img_paths)
        #self.opened_data = np.array(self.opened_data, dtype = float)
        #
        #self.opened_data = self.opened_data.reshape((len(self.img_labels), 1, 64, 64))

       #self.opened_data = np.memmap(self.img_paths, dtype='float32', mode='r', shape = (self.img_labels[index], 3, self.size_of_images, self.size_of_images)).__array__()
        # only one array
        #self.opened_labels = np.load(self.img_paths[:-4] + "_labels.npy")
        #self.opened_labels = np.load(self.img_paths[:-4] + "_labels.npy")
        self.opened_data = np.load(self.img_paths)
        self.opened_data = self.opened_data[:,2,:,:]
        self.opened_data = np.reshape(self.opened_data, (len(self.opened_labels), 1, 64, 64))
        self.input_data = torch.Tensor(self.opened_data)

        tensor = self.input_data[index]#[0]
        #label = np.memmap(self.img_paths, dtype='float32', mode='r', shape = (self.img_labels[index], 3, self.size_of_images, self.size_of_images)).__array__()


        label = self.opened_labels[index]
        #PIL_IMAGE = Image.open(self.img_paths[index]).resize(self.size_of_images)
        #TENSOR_IMAGE = transform(PIL_IMAGE)
        #label = self.img_labels[index]
        return tensor, label

class dataloaders(Dataset):
    
    def __init__(self, paths, labels, testsize_per_channel, datasize = None):
        self.paths = paths
        self.labels = labels
        self.testsize_per_channel = testsize_per_channel
        self.datasize = datasize

    def training(self):
        dataset = CustomDataset(self.paths, self.labels, (self.testsize_per_channel, self.testsize_per_channel))

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                           shuffle=True)
        return self.paths, self.labels, train_loader
    
    def testing(self):
        dataset = CustomDataset(paths,labels,(self.testsize_per_channel, self.testsize_per_channel))
        validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=True)
        return paths_test, labels_test, validation_loader
    


### get dataloaders with input data
image_size = 64




# initialize model
model = model_dict[args.model_file](3)
print(model)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(),lr=1e-4)


TRAIN_LOSS = []
TRAIN_ACCURACY = []

weights_before = []
weights_after = []



interaction_type_map = {
    'nuecc' : 0, 
    'numucc' : 1, 
    'nuenc' : 2, 
    'numunc' : 2
}


for i in interaction_type_map:
    print(i)

if args.random_order:
    random.shuffle(training_data_files)

for epoch in range(1,EPOCH_NUMBER+1):
    epoch_loss = 0.0
    correct = 0
    total = 0
    print(f"Starting epoch #{epoch}")
    for mypath in training_data_files:
        
        mylabels = np.load(mypath)

        current_file =  mypath[:-len("_labels.npy")] + ".npy"
        interaction_data = dataloaders(current_file, mylabels, image_size)
        paths, labels, train_loader = interaction_data.training()
        for data_,target_ in train_loader:
            target_ =target_.to(device)
            data_ = data_.to(device)
            
            # Cleaning the cached gradients if there are
            optimizer.zero_grad()
            
            # Getting train decisions and computing loss.
            outputs = model(data_)
            loss = criterion(outputs,target_)
            
            # Backpropagation and optimizing.
            loss.backward()
            optimizer.step()
            
            # Computing statistics.
            epoch_loss = epoch_loss + loss.item()
            _,pred = torch.max(outputs,dim=1)
            correct = correct + torch.sum(pred == target_).item()
            total += target_.size(0)
        interaction_data = []
    TRAIN_LOSS.append(epoch_loss)
    print(f"Loss for epoch #{epoch}:", epoch_loss)
    TRAIN_ACCURACY.append(100 * correct / total)
    plt.plot(TRAIN_LOSS)
    plt.xlabel("epoch")
    plt.ylabel("Loss value")
    plt.savefig(f"{png_header}_loss_test_dw.png")
    plt.close()
    if epoch == 1 or epoch == 3 or epoch == 10 or epoch == 30 or epoch == 100 or epoch == 200 or epoch == 300:
        print(f"Saving model for epoch #{epoch}")
        torch.save(model, f'./{args.model_file}_{args.model_name}_{epoch}.pt') 




