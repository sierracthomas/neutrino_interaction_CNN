import numpy as np 

def intermediate_points(number_of_points, dE, xf, yf, zf, x0, y0, z0):
    increments = np.linspace(1, number_of_points, number_of_points)
    x = (xf - x0)/(number_of_points + 1)
    y = (yf - y0)/(number_of_points + 1)
    z = (zf - z0)/(number_of_points + 1)
    x = np.array([x * i for i in increments]) + x0
    y = np.array([y * i for i in increments]) + y0
    z = np.array([z * i for i in increments]) + z0
    
    proportional_dE = dE/(number_of_points + 2) 
    dE = [proportional_dE] * (number_of_points + 2)
    
    return np.array([x0, *x, xf]), np.array([y0, *y, yf]), np.array([z0, *z, zf]), dE
    


def define_bin_values_finer_resolution(x, y, z, x0, y0, z0, values, number_of_zbins = 15, x_bins = np.linspace(-65, 65, 130), y_bins = np.linspace(-65, 65, 130), threshold = .150):
    z_bins = np.linspace(-65, 65, number_of_zbins)
    #print(f"height of z-bin is {z_bins[1] - z_bins[0]}")

    test_elements = []
    
    hist, bin_edges = np.histogramdd((x, y, z), bins=[x_bins, y_bins, z_bins])
    x_edges, y_edges, z_edges = bin_edges[0], bin_edges[1], bin_edges[2]
    bin_values = np.zeros_like(hist)

    for i in range(len(x)):
        x_idx = np.digitize(x[i], x_bins) - 1
        y_idx = np.digitize(y[i], y_bins) - 1
        z_idx = np.digitize(z[i], z_bins) - 1
        x0_idx = np.digitize(x0[i], x_bins) - 1
        y0_idx = np.digitize(y0[i], y_bins) - 1
        z0_idx = np.digitize(z0[i], z_bins) - 1  
        
        if x_idx < 0 or x_idx >= bin_values.shape[0]:
            continue
        if y_idx < 0 or y_idx >= bin_values.shape[1]:
            continue
        if z_idx < 0 or z_idx >= bin_values.shape[2]:
            continue
            
            
        ## check if start points and endpoints are different
        ## if so, break up the segments
        
        span_of_data = [x_idx - x0_idx, y_idx - y0_idx, z_idx - z0_idx]
        
        #if x_idx - x0_idx != 0 or y_idx - y0_idx != 0 or z_idx - z0_idx !=0:
            #print(i, x_idx - x0_idx, y_idx - y0_idx, z_idx - z0_idx)
        if any(data != 0 for data in span_of_data):
            number_of_points = np.max(np.abs(span_of_data)) + 1
            l, m, n, energy = intermediate_points(number_of_points, values[i], x[i], y[i], z[i], x0[i], y0[i], z0[i])
            for point in range(len(energy)):
                x_point_idx = np.digitize(l[point], x_bins) - 1
                y_point_idx = np.digitize(m[point], y_bins) - 1
                z_point_idx = np.digitize(n[point], z_bins) - 1

                bin_values[x_point_idx, y_point_idx, z_point_idx] += energy[point]
                test_elements.append([x_point_idx, y_point_idx, z_point_idx])
        else:
            bin_values[x_idx, y_idx, z_idx] += values[i]
            test_elements.append([x_idx, y_idx, z_idx])
            
    unique_nonzero_elements = [list(x) for x in set(tuple(x) for x in test_elements)]
    print("number of elements: ", len(test_elements), "number of unique elements: ", len(unique_nonzero_elements))
    return x_edges, y_edges, z_edges, bin_values, unique_nonzero_elements





def define_bin_values(x, y, z, values, number_of_zbins = 15, x_bins = np.linspace(-65, 65, 130), y_bins = np.linspace(-65, 65, 130)):
    z_bins = np.linspace(-65, 65, number_of_zbins)
    print(f"height of z-bin is {z_bins[1] - z_bins[0]}")

    hist, bin_edges = np.histogramdd((x, y, z), bins=[x_bins, y_bins, z_bins])
    x_edges, y_edges, z_edges = bin_edges[0], bin_edges[1], bin_edges[2]
    bin_values = np.zeros_like(hist)

    for i in range(len(x)):
        x_idx = np.digitize(x[i], x_bins) - 1
        y_idx = np.digitize(y[i], y_bins) - 1
        z_idx = np.digitize(z[i], z_bins) - 1
        if x_idx < 0 or x_idx >= bin_values.shape[0]:
            continue
        if y_idx < 0 or y_idx >= bin_values.shape[1]:
            continue
        if z_idx < 0 or z_idx >= bin_values.shape[2]:
            continue
        bin_values[x_idx, y_idx, z_idx] += values[i]
        
    return x_edges, y_edges, z_edges, bin_values


downsample_dimension = 65


def thresholding_and_downsampling(nonzero_elements, bin_values, x_bins, y_bins, z_bins, threshold_val = 0.150):
    x_dim_midpoint = (x_bins[1] - x_bins[0])/2
    y_dim_midpoint = (y_bins[1] - y_bins[0])/2
    z_dim_midpoint = (z_bins[1] - z_bins[0])/2
    over_threshold = 0
    newx, newy, newz = [], [], []
    newdE = []
    for i in nonzero_elements:
        tempx = x_bins[i[0]] - x_dim_midpoint
        tempy = y_bins[i[1]] - y_dim_midpoint 
        tempz = z_bins[i[2]] - z_dim_midpoint 
        tempdE = bin_values[i[0], i[1], i[2]]
        if tempdE > threshold_val:
            #print(bin_values[i[0], i[1], i[2]])
            newx.append(tempx)
            newy.append(tempy)
            newz.append(tempz)
            newdE.append(tempdE)
            over_threshold += 1
    
    #print(over_threshold, len(nonzero_elements))
    x_edges, y_edges, z_edges, bin_values = define_bin_values(newx, newy, newz, newdE, number_of_zbins = downsample_dimension + 1, x_bins = np.linspace(-65, 65, downsample_dimension + 1), y_bins = np.linspace(-65, 65, downsample_dimension + 1))
    return x_edges, y_edges, z_edges, bin_values
