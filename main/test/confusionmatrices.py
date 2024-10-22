# machine learning libraries
import torch
from torchvision import datasets, transforms
from torch import nn, optim
# import torchvision, which is a complementary tool for image importing in pytorch
import torchvision 
# import numpy for math stuff
import numpy as np
# import matplotlib for nice plotting
import matplotlib.pyplot as plt
# import time so we can see how long training takes!
from time import time
# import seaborn for other nice plotting
import seaborn as sn
# import pandas for data management (particularly for large sets of data)
import pandas as pd

import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Generate confusion matrices', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--testing", type = str, default = "../../datasets/testing/", help = "Testing directory")
parser.add_argument("--training", type = str, default = "../../datasets/training/", help = "Training directory")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()





get_paths = True

if get_paths:
    testing_data_files = []
    training_data_files = []
    # get lengths of arrays in advance!
    for filename in os.listdir(args.testing):
        f = os.path.join(os.getcwd(), args.testing + filename)
        if os.path.isfile(f):
            if f[-len("labels.npy"):] == "labels.npy":
                testing_data_files.append(f)
    for filename in os.listdir(args.training):
        f = os.path.join(os.getcwd(), args.training + filename)
        if os.path.isfile(f):
            if f[-len("labels.npy"):] == "labels.npy":
                training_data_files.append(f)


### get dataloaders with input data

import random
random.shuffle(testing_data_files)
BATCH_SIZE = 128

class CustomDataset(Dataset):
    def __init__(self, img_paths, img_labels, size_of_images, shorten = False):
        
        #self.img_labels = np.load(img_labels)
        self.img_paths = img_paths
        self.size_of_images = size_of_images  
        self.opened_labels = img_labels
        self.shorten = shorten
        #self.opened_labels = torch.Tensor(self.opened_labels)
        #self.size_of_images = size_of_images
        #if len(self.input_data) != len(self.img_labels):
        #    raise InvalidDatasetException(self.img_paths,self.img_labels)
            
    def __len__(self):
        if not self.shorten:
            return len(self.opened_labels)
        if self.shorten:
            return len(self.opened_labels) - self.shorten
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
    
    def __init__(self, paths, labels, testsize_per_channel, datasize = None, shorten = None):
        self.paths = paths
        self.labels = labels
        self.testsize_per_channel = testsize_per_channel
        self.datasize = datasize
        self.shorten = shorten

    def training(self):
        dataset = CustomDataset(self.paths, self.labels, (self.testsize_per_channel, self.testsize_per_channel), shorten = self.shorten)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                           shuffle=True)
        return self.paths, self.labels, train_loader
    
    


### get dataloaders with input data
image_size = 64



def eval_for_confmat(validation_loader, model):
    total_val_loss = 0.0
    total_true = 0
    total = 0
    actual = []
    predicted = []

    # When we're not working with gradients and backpropagation
    # we use torch.no_grad() utility.
    with torch.no_grad():
        model.eval()
        for data_,target_ in validation_loader:
            total += 1
            data_ = data_.to(device)
            target_ = target_.to(device)

            outputs = model(data_)
            loss = criterion(outputs,target_).item()
            _,preds = torch.max(outputs,dim=1)
            total_val_loss += loss
            true = torch.sum(preds == target_).item()
            #print(preds)
            predicted.append(np.array(preds))
            #print(target_)
            actual.append(np.array(target_))
            total_true += true

    validation_accuracy = round(100 * total_true / total,2)
    #print(f"Validation accuracy: {validation_accuracy}%")
    #print(f"Validation loss: {round(total_val_loss,2)}%")
    return actual, predicted




# compute confusion matrix 
def comp_confmat(actual, predicted):
    actual = np.hstack(actual)
    predicted = np.hstack(predicted)
    # extract the different classes
    classes = np.unique(actual)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):

           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat

interaction_type_map = {
    'nuecc' : 0, 
    'numucc' : 1, 
    'nuenc' : 2, 
    'numunc' : 2
}


def plot_confusion_matrix(confusion_matrix, savepic):
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in ['nuecc', 'numucc', 'nc']],
                  columns = [i for i in ['nuecc', 'numucc', 'nc']])
    plt.figure(figsize = (10,7))


    ax = sn.heatmap(df_cm, annot=True,cmap="OrRd")
    ax.set(ylabel="Truth", xlabel="Predicted")
    plt.suptitle(f"Confusion matrix of model on {np.sum(confusion_matrix)} tests")
    ax.xaxis.tick_top()
    plt.savefig(savepic)
    plt.close()
    


from fpdf import FPDF
from PIL import Image
import glob
import os


import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


current_model_list = []
for infile in sorted(glob.glob('*.pt'), key=numericalSort):
    print("Current File Being Processed is: ", infile)
    current_model_list.append(infile)

for current_model in current_model_list:
    #current_model = "modeldw_run_number_1_30.pt"
    model = torch.load(current_model)
    optimizer = optim.RMSprop(model.parameters(),lr=1e-4)
    print(model)
    allactual, allpredicted = [], []
    for mypath in testing_data_files:
        mylabels = np.load(mypath)
        prev_label = mylabels[0]

        for inttype in interaction_type_map:
            if inttype in mypath:
                adjusted_label = interaction_type_map[inttype]
                continue
        mylabels = [adjusted_label] * len(mylabels)

        print(f"Changing labels for {mypath} from {prev_label} to {mylabels[0]}")
        print(len(mylabels))
        current_file =  mypath[:-len("_labels.npy")] + ".npy"
        if "nuenc_h5.0000009_1_labels.npy" in mypath:
            #mylabels = mylabels[:-19]
            interaction_data = dataloaders(current_file, mylabels, image_size, shorten = 19)
        else:
            interaction_data = dataloaders(current_file, mylabels, image_size)
        paths, labels, train_loader = interaction_data.training()
        actual, predicted = eval_for_confmat(train_loader, model)
        allactual.append(np.hstack(actual))
        allpredicted.append(np.hstack(predicted))
    conf = comp_confmat(np.hstack(allactual), np.hstack(allpredicted))
    print("Unique labels: ", np.unique(np.hstack(allpredicted)))
    plot_confusion_matrix(conf, f"{str(current_model)[:-3]}_confmat.png")
    

exit()
