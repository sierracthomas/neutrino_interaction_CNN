
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Train a convolutional neural network for neutrino interactions. ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", type = int, default = 5, help = "Set random seed")
#parser.add_argument("--image_dir", type = str, default = "/home/sthoma31/neutrino_interaction_images/nu_mu_700/data/QES", nargs='*', help = "Location that contains interaction folders CC and NC")
parser.add_argument("--numu_folder", type = str, default = "/home/sthoma31/neutrino_interaction_images/array_generator/numu_10_7", nargs='*', help = "Name of folder containing Nu_mu interactions")
parser.add_argument("--nue_folder", type = str, default = "/home/sthoma31/neutrino_interaction_images/array_generator/nue_10_7", nargs='*', help = "Name of folder containing Nue interactions")
parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size when training")
#parser.add_argument("--training_data_size", type = int, default = 4000, help = "Size of training data")
parser.add_argument("--epoch", type = int, default = 20, help = "Number of epochs when training")
parser.add_argument("--testing_data_size", type = int, default = 100, help = "Size of testing data for each channel ")
parser.add_argument("--png_header", type = str, default = "trial", help = "Header name for PNG files")
parser.add_argument("--plot_freq", type = int, default = 5, help = "Plot confusion matrices every {plot_freq} times")
parser.add_argument("--model_name", type = str, default = "run_number_1", help = "Name of model to save as")
parser.add_argument("--model_file", type = str, default = "modeldw", help = "Model to use to train")
parser.add_argument("--move_files", type = bool, default = True, help = "Sort into testing and training files")
parser.add_argument("--training_size", type = int, default = 8000, help = "Approx. size of training dataset")

args = parser.parse_args()
torch.manual_seed(args.seed)
#IMAGE_LOC = args.image_dir
BATCH_SIZE = args.batch_size
EPOCH_NUMBER = args.epoch
#datasize = args.training_data_size
testsize_per_channel = args.testing_data_size
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

    def training_testing_data(total_events = args.training_size, interaction_shape_list = numucc_shape_list, interaction_file_list = numucc_file_list):
        total_events = total_events + 1
        elements_in_total_events = 0
        print("Total interactions: ", np.sum(interaction_shape_list))
        for i in interaction_shape_list:
            total_events = total_events - i
            if total_events < 0:
                break
            elif total_events > 0:
                elements_in_total_events = elements_in_total_events + 1
        return interaction_file_list[:elements_in_total_events], interaction_file_list[elements_in_total_events:]

    training_numucc, testing_numucc = training_testing_data(interaction_shape_list = numucc_shape_list, interaction_file_list = numucc_file_list)
    training_numunc, testing_numunc = training_testing_data(total_events = int(args.training_size/2), interaction_shape_list = numunc_shape_list, interaction_file_list = numunc_file_list)

    training_nuecc, testing_nuecc = training_testing_data(interaction_shape_list = nuecc_shape_list, interaction_file_list = nuecc_file_list)
    training_nuenc, testing_nuenc = training_testing_data(total_events = int(args.training_size/2), interaction_shape_list = nuenc_shape_list, interaction_file_list = nuenc_file_list)

    training_data_files = training_nuenc + training_nuecc + training_numunc + training_numucc
    testing_data_files = testing_numucc + testing_numunc + testing_nuecc + testing_nuenc
    print(len(training_data_files), len(testing_data_files))

    #print(training_data_files[0:10])
    #print(np.sum(numucc_shape_list[:elements_in_total_events]))
    #print("numunc", numunc_shape_list, np.sum(numunc_shape_list))
    #print("nuecc", nuecc_shape_list, np.sum(nuecc_shape_list))
    #print("nuenc", nuenc_shape_list, np.sum(nuenc_shape_list))\
    import shutil
    if os.path.exists("testing/"):
        shutil.rmtree("testing/")

    if os.path.exists("training/"):
        shutil.rmtree("training/")
    if not os.path.exists("testing/"):
        os.makedirs("testing/")
    if not os.path.exists("training/"):
        os.makedirs("training/")

    for i in training_data_files:
        shutil.copy(i, 'training/')
        shutil.copy(i[:-len("_labels.npy")] + ".npy", 'training/')

    for i in testing_data_files:
        shutil.copy(i, 'testing/')
        shutil.copy(i[:-len("_labels.npy")] + ".npy", 'testing/')



elif not args.move_files:
    testing_data_files = []
    training_data_files = []
    # get lengths of arrays in advance!
    for filename in os.listdir("testing/"):
        f = os.path.join(os.getcwd(), "testing/" + filename)
        if os.path.isfile(f):
            if f[-len("labels.npy"):] == "labels.npy":
                testing_data_files.append(f)
    for filename in os.listdir("training/"):
        f = os.path.join(os.getcwd(), "training/" + filename)
        if os.path.isfile(f):
            if f[-len("labels.npy"):] == "labels.npy":
                training_data_files.append(f)


### get dataloaders with input data

import random
random.shuffle(training_data_files)

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

import matplotlib.pyplot as plt

dataset_number = 0
"""
for mypath in nuenc_file_list:
    mylabels = np.load(mypath)
    print(len(mylabels))
    current_file =  mypath[:-11] + ".npy"
    interaction_data = dataloaders(current_file, mylabels, image_size)
    paths, labels, train_loader = interaction_data.training()
    print(len(paths))

    images, labels = next(iter(train_loader))

    fig, axis = plt.subplots(5, 5, figsize=(15, 10))
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            npimg = images[i].numpy()

            npimg = np.transpose(npimg, (2, 1, 0))
            label = int(labels[i])
            ax.imshow(npimg, cmap = "Greys_r")
            ax.set(title = f"{label}")

    plt.savefig(f"test_{dataset_number}.png")
    plt.close()
    dataset_number += 1
    break"""




# initialize model
model = model_dict[args.model_file](3)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(),lr=1e-4)


TRAIN_LOSS = []
TRAIN_ACCURACY = []

weights_before = []
weights_after = []

location = 0

EPOCH_NUMBER = 200
interaction_type_map = {
    'nuecc' : 0, 
    'numucc' : 1, 
    'nuenc' : 2, 
    'numunc' : 2
}


for i in interaction_type_map:
    print(i)
import random
random.shuffle(training_data_files)

for epoch in range(1,EPOCH_NUMBER+1):
    epoch_loss = 0.0
    correct = 0
    total = 0
    print(f"Starting epoch #{epoch}")
    for mypath in training_data_files:
        
        mylabels = np.load(mypath)
        prev_label = mylabels[0]

        for inttype in interaction_type_map:
            if inttype in mypath:
                adjusted_label = interaction_type_map[inttype]
                continue

        mylabels = [adjusted_label] * len(mylabels)
        print(f"Changing labels for {mypath} from {prev_label} to {mylabels[0]}")
        current_file =  mypath[:-len("_labels.npy")] + ".npy"
        interaction_data = dataloaders(current_file, mylabels, image_size)
        paths, labels, train_loader = interaction_data.training()
        for data_,target_ in train_loader:
            #print(location, conv1.weight.data[0])
            location +=1
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
    
    if epoch == 1 or epoch == 3 or epoch == 10 or epoch == 30 or epoch == 100 or epoch == 200 or epoch == 300:
        print(f"Saving model for epoch #{epoch}")
        torch.save(model, f'./{args.model_file}_{args.model_name}_{epoch}.pt') 
        #with open(f"{args.model_file}_{args.model_name}_{epoch}.txt", "w") as file_info:
        #    for i in nuenc_file_list:
        #        file_info.write(i + '/n')
        # save file, save model
        # make a file to load model, testing data
        # print all these to have a snapshot as it's training 
        # number of events trained and tested on 

        # train on 30,000 and while it's training then work on the confusion matrix

    #print(f"Epoch {epoch}: Accuracy: {100 * correct/total}, Loss: {epoch_loss}")
    #if epoch % plot_frequency == 0:
    #    actual, predicted = eval_for_confmat(validation_loader, model = model)
    #    confmat = comp_confmat(actual, predicted)
    #    plot_confusion_matrix(confmat, f"{png_header}_{epoch}.png")
    #    torch.save(model, f"./epoch_{epoch}.pt")
    #    model.train()


# In[10]:



import matplotlib
matplotlib.rcParams.update({'font.size': 22})
plt.plot(TRAIN_LOSS)
plt.xlabel("epoch")
plt.ylabel("Loss value")
#plt.yscale("log")

plt.savefig("loss_test_dw.png")
plt.close()

# In[11]:


print(TRAIN_LOSS)
exit()


# In[ ]:


test = np.load("all_arrays.npy")
test = np.array(test, dtype = float)
print(test.shape)
test = test[:,2,:,:] # only beam images
test = test.reshape((7, 1, 64, 64)) # add extra dimension of 1 so torch doesn't complain

input_data = torch.Tensor(test)
with torch.no_grad():
    print(model(input_data))


# In[ ]:


before_conv1 = weights[0]

fig, axis = plt.subplots(4,4, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = before_conv1[i][0].numpy()
        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = before_conv1.min(), vmax = before_conv1.max())
        #ax.set(title = f"{label}")
plt.subplots_adjust(wspace=0, hspace=0)        
plt.show()

class CustomDataset(Dataset):
    def __init__(self, img_paths, img_labels, size_of_images):
        self.img_labels = np.load(img_labels)
        self.img_paths = img_paths
        self.opened_data = np.load(img_paths)
        self.opened_data = np.array(self.opened_data, dtype = float)
        self.opened_data = self.opened_data[:,2,:,:]
        self.opened_data = self.opened_data.reshape((len(self.img_labels), 1, 64, 64))
        self.input_data = torch.Tensor(self.opened_data)
        
        
        self.size_of_images = size_of_images
        if len(self.input_data) != len(self.img_labels):
            raise InvalidDatasetException(self.img_paths,self.img_labels)
            
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, index):
        tensor = self.input_data[index]#[0]
        label = self.img_labels[index]
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
                                           shuffle=False)
        return self.paths, self.labels, train_loader
    
    def testing(self):
        dataset = CustomDataset(paths,labels,(self.testsize_per_channel, self.testsize_per_channel))
        validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=True)
        return paths_test, labels_test, validation_loader
    
testsize_per_channel = 128

### get dataloaders with input data



interaction_data = dataloaders("/Users/sierra/Dropbox/neutrino_work/summer_2024/ml/scratchwork_energy_averaging/ml_with_arrays/nu_mu_700_testtrial.npy", "/Users/sierra/Dropbox/neutrino_work/summer_2024/ml/scratchwork_energy_averaging/ml_with_arrays/nu_mu_700_testtrial_labels.npy", testsize_per_channel)

paths, labels, train_loader = interaction_data.training()

images, labels = next(iter(train_loader))
#images = iter(validation_loader)
#print(images.shape)
fig, axis = plt.subplots(1, 7, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        print(model(images))
        npimg = images[i].numpy()
        print(npimg.shape)

        npimg = np.transpose(npimg, (2, 1, 0))
        #label = label_map[int(labels[i])]
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")


# In[34]:


class CustomDataset(Dataset):
    def __init__(self, img_paths, img_labels, size_of_images):
        self.img_labels = np.load(img_labels)
        self.img_paths = img_paths
        self.opened_data = np.load(img_paths)
        self.opened_data = np.array(self.opened_data, dtype = float)
        self.opened_data = self.opened_data[:,1,:,:]
        self.opened_data = self.opened_data.reshape((len(self.img_labels), 1, 64, 64))
        self.input_data = torch.Tensor(self.opened_data)
        
        
        self.size_of_images = size_of_images
        if len(self.input_data) != len(self.img_labels):
            raise InvalidDatasetException(self.img_paths,self.img_labels)
            
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, index):
        tensor = self.input_data[index]#[0]
        label = self.img_labels[index]
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
                                           shuffle=False)
        return self.paths, self.labels, train_loader
    
    def testing(self):
        dataset = CustomDataset(paths,labels,(self.testsize_per_channel, self.testsize_per_channel))
        validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=True)
        return paths_test, labels_test, validation_loader
    
testsize_per_channel = 128

interaction_data = dataloaders("/Users/sierra/Dropbox/neutrino_work/summer_2024/ml/scratchwork_energy_averaging/ml_with_arrays/nu_mu_700_testtrial.npy", "/Users/sierra/Dropbox/neutrino_work/summer_2024/ml/scratchwork_energy_averaging/ml_with_arrays/nu_mu_700_testtrial_labels.npy", testsize_per_channel)

paths, labels, train_loader = interaction_data.training()

def eval_for_confmat(validation_loader, model = model):
    total_val_loss = 0.0
    total_true = 0

    actual = []
    predicted = []

    # When we're not working with gradients and backpropagation
    # we use torch.no_grad() utility.
    with torch.no_grad():
        model.eval()
        for data_,target_ in validation_loader:
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

actual, predicted = eval_for_confmat(train_loader)


# In[35]:


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

conf = comp_confmat(actual, predicted)


# In[36]:


def plot_confusion_matrix(confusion_matrix, savepic):
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in ['CC', 'NC']],
                  columns = [i for i in ['CC', 'NC']])
    plt.figure(figsize = (10,7))


    ax = sn.heatmap(df_cm, annot=True,cmap="OrRd")
    ax.set(ylabel="Truth", xlabel="Predicted")
    plt.suptitle(f"Confusion matrix of model on {np.sum(confusion_matrix)} tests")
    ax.xaxis.tick_top()
    plt.savefig(savepic)
    plt.show()
    
plot_confusion_matrix(conf, "/Users/sierra/Downloads/confmatrix.png")
    


# In[30]:



model = nn.Sequential(*model_list[0:1])
print("Model: \n", model)
before_conv1 = conv1.weight.data
conv_tensor = model(images[0])
print(conv_tensor.shape)
after_conv1 = conv1.weight.data
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


"""# plot on same axis
fig, axis = plt.subplots(1, 1, figsize=(25, 20))
for i in range(0, 1):
    with torch.no_grad():
        npimg = [torch.cat([conv_tensor[i] for i in range(j * 8, 8 * (j + 1))]) for j in range(0, 4)]#conv_tensor[i].numpy()
        npimg = torch.cat(npimg, dim =1)
        npimg = np.transpose(npimg, (1, 0))
        plt.imshow(npimg, cmap = "Greys")

plt.show()"""
# alternatively, plot on different axes

fig, axis = plt.subplots(4,4, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[i].numpy()
        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")
plt.subplots_adjust(wspace=0, hspace=0)        
plt.show()

"""fig, axis = plt.subplots(4, 1, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    loop = (i * 8, 8*(i+1))
    print(loop)
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = torch.cat([conv_tensor[i] for i in range(*loop)])#conv_tensor[i].numpy()
        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")"""


"""fig, axis = plt.subplots(4, 8, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[i].numpy()
        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")
"""


# In[ ]:


fig, axis = plt.subplots(4,4, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = before_conv1[i][0].numpy()
        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = before_conv1.min(), vmax = before_conv1.max())
        #ax.set(title = f"{label}")
plt.subplots_adjust(wspace=0, hspace=0)        
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:2])
images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)

#conv_tensor = model(images[0])
print("Model: \n", model)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


"""# plot on same axis
fig, axis = plt.subplots(1, 1, figsize=(25, 20))
for i in range(0, 1):
    with torch.no_grad():
        npimg = [torch.cat([conv_tensor[i] for i in range(j * 8, 8 * (j + 1))]) for j in range(0, 8)]#conv_tensor[i].numpy()
        npimg = torch.cat(npimg, dim =1)
        npimg = np.transpose(npimg, (1, 0))
        plt.imshow(npimg, cmap = "Greys")


plt.show()"""


fig, axis = plt.subplots(4, 4, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()
        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:4])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 8, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:5])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 8, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:6])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 8, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:7])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 8, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:8])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 8, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:9])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 8, figsize=(25, 20))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys", vmin = conv_tensor.min(), vmax = conv_tensor.max())
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:10])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 4, figsize=(25, 20))
for i, ax in enumerate(axis.flatten()):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:11])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 4, figsize=(25, 20))
for i, ax in enumerate(axis.flatten()):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:12])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 4, figsize=(25, 20))
for i, ax in enumerate(axis.flatten()):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:13])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 4, figsize=(25, 20))
for i, ax in enumerate(axis.flatten()):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:14])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(4, 4, figsize=(25, 20))
for i, ax in enumerate(axis.flatten()):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor[0][i].numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        ax.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:15])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(1, 1, figsize=(25, 20))
for i in range(0, 1):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor.numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        axis.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list[0:16])
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(1, 1, figsize=(25, 20))
for i in range(0, 1):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor.numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        axis.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:


model = nn.Sequential(*model_list)
print("Model: \n", model)

images_batch = np.reshape(images[0:1], (1, 1, 64, 64))
conv_tensor = model(images_batch)
print(conv_tensor.shape)
npimg = np.transpose(images[0], (2, 1, 0))
plt.imshow(npimg, cmap = "Greys")
plt.show()


fig, axis = plt.subplots(1, 1, figsize=(25, 20))
for i in range(0, 1):
    with torch.no_grad():
        #images = next(images)
        #images = next(iter(validation_loader))
        
        npimg = conv_tensor.numpy()

        #npimg = np.transpose(images[0], (2, 1, 0))
        npimg = np.transpose(npimg, (1, 0))
        axis.imshow(npimg, cmap = "Greys")
        #ax.set(title = f"{label}")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# In[ ]:





# In[ ]:


# pool of square window of size=3, stride=2
#m = nn.MaxPool2d(3, stride=2)
# pool of non-square window


# # step examples below:

# In[ ]:


m = nn.MaxPool2d((3, 3), stride=(2, 2))
inp = torch.randn(1, 1, 5, 5)
output = m(inp)
print(inp.max(), inp.min(), output.max(), output.min())


print("input: ", inp.shape)
plt.imshow(inp[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()
print("output: ", output.shape)
plt.imshow(output[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()


# In[ ]:




m = nn.BatchNorm2d(2, affine=False)

inp = torch.randn(1, 2, 2, 2)
output = m(inp)
#print(inp.max(), inp.min(), output.max(), output.min())


print("input: ", inp.shape, "max: ", inp[0][0].max(), "min: ", inp[0][0].min())
plt.imshow(inp[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()
print("input: ", inp.shape, "max: ", inp[0][1].max(), "min: ", inp[0][1].min())
plt.imshow(inp[0][1], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()

print("output: ", output.shape, "max: ", output[0][0].max(), "min: ", output[0][0].min())
plt.imshow(output[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()

print("output: ", output.shape, "max: ", output[0][1].max(), "min: ", output[0][1].min())
plt.imshow(output[0][1], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()


# In[ ]:


torch.Tensor(np.identity(2)) + 1


# In[ ]:




m = nn.LogSoftmax(dim=2)

identity = np.identity(2) + 1.
identity = np.reshape(identity, (1,1,2,2))
inp = torch.Tensor(identity)
output = m(inp)
#print(inp.max(), inp.min(), output.max(), output.min())


print("input: ", inp.shape, "max: ", inp[0][0].max(), "min: ", inp[0][0].min())
plt.imshow(inp[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()

print("output: ", output.shape, "max: ", output[0][0].max(), "min: ", output[0][0].min())
plt.imshow(output[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()


# In[ ]:


m = nn.Linear(2, 3)

identity = np.identity(2)
identity = np.reshape(identity, (1,1,2,2))
inp = torch.Tensor(identity)
output = m(inp)
#print(inp.max(), inp.min(), output.max(), output.min())


print("input: ", inp.shape, "max: ", inp[0][0].max(), "min: ", inp[0][0].min())
plt.imshow(inp[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()


output = output.detach().numpy()
print("output: ", output.shape, "max: ", output[0][0].max(), "min: ", output[0][0].min())
plt.imshow(output[0][0], cmap = "Greys", vmin = output.min(), vmax = inp.max() + 0.5)
plt.show()


# In[ ]:





# In[ ]:





