import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from electron_conversion import *
from define_bin_values import * 
import argparse
import os
import sys


parser = argparse.ArgumentParser(description='Generate arrays for a dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--directory", type = str, default = ".", help = "Directory to access files")
parser.add_argument("--out_directory", type = str, default = ".",  help = "Directory to save arrays")
parser.add_argument("--interaction_type", type = str, choices = ['nuecc', 'nuenc', 'numucc', 'numunc'], help = "Interaction type to generate")
parser.add_argument("--length_in_filename", type = bool, default = False, help = "When set to true, add length of file in name")

args = parser.parse_args()
save_after_n_interactions = 500
desired_interaction = args.interaction_type
working_directory = args.directory
out_directory = args.out_directory

file_list = os.listdir(working_directory)
hdf5_list = [i for i in file_list if i[-5:] == ".hdf5"]


for i in hdf5_list:
    print(i)



genie_reaction_map = {
    "QES" : 1,
    "1Kaon" : 2,
    "DIS" : 3,
    "RES" : 4,
    "COH" : 5,
    "DFR" : 6,
    "NuEEL" : 7,
    "IMD": 8,
    "AMNuGamma": 9,
    "MEC": 10,
    "CEvNS": 11,
    "IBD": 12,
    "GLR": 13,
    "IMDAnh": 14,
    "PhotonCOH": 15,
    "PhotonRES": 16,
    "1Pion": 17,
    "DMEL": 101,
    "DMDIS": 102,
    "DME": 103,
}

mc_hdr = ['event_id','vertex_id','x_vert','y_vert','z_vert','t_vert','target','reaction','isCC','isQES','isMEC','isRES','isDIS','isCOH','Enu','nu_4mom','nu_pdg','Elep','lep_mom','lep_ang','lep_pdg','q0','q3','Q2','x','y']

inverse_reaction_map = {v: k for k, v in genie_reaction_map.items()}

interaction_type_map = {
    'nuenc' : 0, 
    'nuecc' : 1, 
    'numunc' : 2, 
    'numucc' : 3
}
    
# check if interaction is nc or cc
check_type = {
    'nuecc' : lambda num: myfile['mc_hdr'][num][8] == True, 
    'nuenc' : lambda num: myfile['mc_hdr'][num][8] != True, 
    'numucc' : lambda num: myfile['mc_hdr'][num][8] == True, 
    'numunc' : lambda num: myfile['mc_hdr'][num][8] != True, 
}


# check if particle is nue or numu
check_particle = {
    'nuecc' : lambda j: myfile['mc_hdr'][j]['nu_pdg'] == 12, 
    'nuenc' : lambda j: myfile['mc_hdr'][j]['nu_pdg'] == 12, 
    'numucc' : lambda j: myfile['mc_hdr'][j]['nu_pdg'] == 14, 
    'numunc' : lambda j: myfile['mc_hdr'][j]['nu_pdg'] == 14, 
}

for current_file in hdf5_list:
    filename = f"{working_directory}/{current_file}"
    print("Starting file :", filename)
    myfile = h5py.File(filename, "r")
    segbreaks = []

    label_list = []
    save_label = interaction_type_map[desired_interaction]

    vstart = 0
    segbreaks.append(0)

    # find where events start and end
    print("Counting events...")
    for i, val in enumerate(myfile['segments']['event_id']):
        if val - vstart > 0:
            ##print(i)
            segbreaks.append(i)
            vstart = val
            ##print("event", vstart)
            continue

    print(len(segbreaks), " events to generate.")        

    record_de = []

    print("Making images...")

    single_bin_values = []

    test_x, test_y, test_z = [], [], []

    j = 0
    
    # files saved already in this batch
    number_of_save_files = 0
    if not check_particle[desired_interaction](j):
        print("Wrong particle type. ")
        sys.exit()

    print("range: ", range(len(segbreaks) - 1))
    last_event_in_file = range(len(segbreaks) - 1)[-1]
    for num in range(len(segbreaks) - 1): 
        
        print(num, "last number: ", last_event_in_file)
        print(inverse_reaction_map[myfile['mc_hdr'][j][7]])

        print(f"event is {myfile['mc_hdr'][j][0]}")
        print(f"pdg is {myfile['mc_hdr'][j]['nu_pdg']}")
        
        if num != last_event_in_file:
        # if interaction is not QES and not the last one in file, keep going
            if inverse_reaction_map[myfile['mc_hdr'][j][7]] != "QES": 
                j += 1
                print("Skipping non-QES...")
                print("-------------------- \n")
                continue

            # if interaction is wrong and not the last one in file, keep going
            if not check_type[desired_interaction](num): 
                print(f"Not {desired_interaction}, skipping...")
                print("-------------------- \n")
                j += 1
                continue

        # if this is the last event in file, save previous events that meet specifications
        elif num == last_event_in_file:
            # just in case there's no desired interaction, don't try to save anything 
            if len(label_list) == 0:
                continue
            if inverse_reaction_map[myfile['mc_hdr'][j][7]] != "QES" or not check_type[desired_interaction](num):
                print("Last event, not desired interaction ", len(label_list))

                print(len(label_list), "Saving...")

                #save file 
                arrays_to_write =  starting_array#[1:]
                print("SHAPE: ", arrays_to_write.shape, len(label_list))
                arrays_to_write = np.reshape(arrays_to_write, (len(label_list), 3, downsample_dimension -1, downsample_dimension-1))
                print("RESHAPE: ", arrays_to_write.shape, len(label_list))

                np.save(f"{out_directory}/{desired_interaction}_{current_file[-23:-13]}_{number_of_save_files}.npy", arrays_to_write)
                np.save(f"{out_directory}/{desired_interaction}_{current_file[-23:-13]}_{number_of_save_files}_labels.npy", label_list)

                print(f"{out_directory}/{desired_interaction}_{current_file[-23:-13]}_{number_of_save_files}.npy")
                number_of_save_files += 1 
                print("-------------------- \n")
                # reset_label_list
                label_list = []
                # continue j iteration
                j += 1
                continue
        for k in range(7,15):
            print(mc_hdr[k], myfile['mc_hdr'][j][k])

        x = myfile['segments']['x_end'][segbreaks[num]:segbreaks[num + 1]]
        y = myfile['segments']['y_end'][segbreaks[num]:segbreaks[num + 1]]
        z = myfile['segments']['z_end'][segbreaks[num]:segbreaks[num + 1]]

        x0 = myfile['segments']['x_start'][segbreaks[num]:segbreaks[num + 1]]
        y0 = myfile['segments']['y_start'][segbreaks[num]:segbreaks[num + 1]]
        z0 = myfile['segments']['z_start'][segbreaks[num]:segbreaks[num + 1]]
        dE = myfile['segments']['dE'][segbreaks[num]:segbreaks[num + 1]]

        record_de.append(dE)
        
        ## 2 cm by 2 cm pixels by 4 mm depth
        x_slot, xbins = 'Drift', 8125 + 1
        y_slot, ybins = 'Vertical', 260 + 1
        slice_view, number_of_z_bins = 'Beam', 260 + 1

        # detector edges with number of bins
        x_bins = np.linspace(-65, 65, xbins)
        y_bins = np.linspace(-65, 65, ybins)

        drift, vertical, beam, bin_values, nonzero_elements = define_bin_values_finer_resolution(x, y, z, x0, y0, z0, dE, number_of_zbins = number_of_z_bins, x_bins = x_bins, y_bins = y_bins)

        # try this to reduce memory issues:
        x, x0, y, y0, z, z0, dE = [], [], [], [], [], [], []

        z_bins = np.linspace(-65, 65, number_of_z_bins)
        drift, vertical, beam, bin_values = thresholding_and_downsampling(nonzero_elements, bin_values, drift, vertical, beam) 

        # cutoff one bin from each dimension to get 64 bins with 2 cm each
        bin_values = bin_values[1:,1:,1:]
        drift, vertical, beam = drift[1:], vertical[1:], beam[1:]

        beam_projected = np.sum(bin_values, axis = 2).T
        vertical_projected = np.sum(bin_values, axis = 1)
        drift_projected = np.sum(bin_values, axis = 0)
        #single_bin_values.append(beam_projected)

        if len(label_list) == 0:
            starting_array = np.zeros((downsample_dimension -1, downsample_dimension -1))
            starting_array = np.append([starting_array], [beam_projected, vertical_projected, drift_projected], axis = 0)
            starting_array = starting_array[1:]

        elif len(label_list) > 0:
            starting_array = np.append(starting_array, [beam_projected, vertical_projected, drift_projected], axis = 0)


        label_list.append(save_label)
        print("Appended values, now labels are: ", len(label_list))

        print("-------------------- \n")


        if len(label_list) == 0:
            # stuff mod 0 behaves funny
            j += 1
            continue

        # if labels show enough files to save or if we're in the last iteration
        elif num == last_event_in_file or len(label_list) % save_after_n_interactions == 0:
            print("Last event, desired interaction ", len(label_list))
            
            print(len(label_list), "Saving...")

            #save file 
            arrays_to_write =  starting_array#[1:]
            print("SHAPE: ", arrays_to_write.shape, len(label_list))
            arrays_to_write = np.reshape(arrays_to_write, (len(label_list), 3, downsample_dimension -1, downsample_dimension-1))
            print("RESHAPE: ", arrays_to_write.shape, len(label_list))

            np.save(f"{out_directory}/{desired_interaction}_{current_file[-23:-13]}_{number_of_save_files}.npy", arrays_to_write)
            np.save(f"{out_directory}/{desired_interaction}_{current_file[-23:-13]}_{number_of_save_files}_labels.npy", label_list)

            print(f"{out_directory}/{desired_interaction}_{current_file[-23:-13]}_{number_of_save_files}.npy")
            number_of_save_files += 1 
            print("--------------------")
            # reset_label_list
            label_list = []
            arrays_to_write = []
            starting_array = []
            # continue j iteration
            j += 1
            continue
        elif len(label_list) % save_after_n_interactions != 0:
            j += 1
            continue
