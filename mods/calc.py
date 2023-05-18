from scipy.ndimage import label
import pyvista as pv
import pandas as pd
import os
import numpy as np
import utils.binvox_rw
from utils import *

import mods

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
 
    return os.path.join(base_path, relative_path)


def invert_binvox(binvox_data):
    # Extract the shape and data from the binvox object
    data = binvox_data.data
    
    # Invert the data by using the bitwise XOR operator
    inverted_data = np.ones(data.shape, dtype=bool) ^ data
    
    # Return the inverted binvox object
    return inverted_data

def volcalc_piece(filename,scale):
    factor=3.7774124910572735e-06#to convert binvox unit to mm^3
    # Load the binvox file using binvox_rw
    with open(resource_path(r'Staging Area\\'+filename+".binvox"), "rb") as f:
        model = utils.binvox_rw.read_as_3d_array(f)

    data=model.data
    data = invert_binvox(model)
    #data=data.astype(int)
    #data=dense_to_sparse(data)

    # Convert the binvox data into a numpy array
    #data = np.array(model.data, dtype=bool)


    # Label each connected component in the binvox data
    labeled_data, num_components = label(data)
    
    calc_info=np.empty((0, 5))
    vol=[]
    # Loop over each labeled component and calculate its volume and centroid
    for i in range(1, num_components+1):
        # Extract the binary mask for this component
        component = labeled_data == i

        # Calculate the volume of the component
        volume = np.sum(component) *factor* scale**3#model.scale[0] ** 3

        # Calculate the centroid of the component
        positions = np.argwhere(component)
        centroid = np.mean(positions, axis=0) * scale#model.scale[0]
        
        vol.append(volume)
        
        
        centroid=np.append(centroid,volume)
        # Print the results for this component
        #print(f"Component {i}: volume={volume/(1)}, centroid={centroid}")
        calc_info = np.vstack((calc_info,np.append(centroid,-1)))
        
    return vol,calc_info

def volcalc_whole(filename,scale):
    factor=3.7774124910572735e-06#to convert binvox unit to mm^3
    # Load the binvox file using binvox_rw
    with open(resource_path(r'Staging Area\\'+filename+".binvox"), "rb") as f:
        model = utils.binvox_rw.read_as_3d_array(f)

    data=model.data
    data = invert_binvox(model)
    #data=data.astype(int)
    #data=dense_to_sparse(data)

    # Convert the binvox data into a numpy array
    #data = np.array(model.data, dtype=bool)


    # Label each connected component in the binvox data
    labeled_data, num_components = label(model.data)

    
    
    # Loop over each labeled component and calculate its volume and centroid
    for i in range(1, num_components+1):
        # Extract the binary mask for this component
        component = labeled_data == i

        # Calculate the volume of the component
        volume = np.sum(component)*factor * scale**3#model.scale[0] ** 3

        # Calculate the centroid of the component
        positions = np.argwhere(component)
        centroid = np.mean(positions, axis=0) * scale#model.scale[0]

        
        # Print the results for this component
        #print(f"Component {i}: volume={volume/(1)}, centroid={centroid}")
        
    return volume
    
def grpby(arr):
    # get unique values from the first column
    unique_vals = np.unique(arr[:, 0])

    # loop over the unique values and add up the values of the second column
    result = []
    for val in unique_vals:
        mask = arr[:, 0] == val
        sum_val = np.sum(arr[mask, 1])
        result.append([val, sum_val])

    # convert result to a NumPy array
    result = np.array(result)
    
    return result

def get_df(filename,labels,tool,rate,speed,scale):

    array1 = np.array([12.7, 22, 551.68, 0.4, 220.67])
    array2 = np.array([12, 20, 530.79, 0.12, 63.69])

    # create the base array
    tool = np.full((24, 5), [1, 1, 1, 1, 1])

    tool[1::2, :] = np.array([12.7,22,551.68,0.4,220.67])

    # Assign odd rows with [12,20,530.79,0.12,63.69]
    tool[::2, :] = np.array([12,20,530.79,0.12,63.69])

    #tool = np.full((24, 5), [1, 1,1,1,1])
    tool[19]=np.array([12,20,530.79,0.12,63.69])
    tool[2]=np.array([12.7,22,551.68,0.4,220.67])
    tool[1]=np.array([12.7,22,551.68,0.4,220.67])

    rate=np.ones([24,1])*7

    speed=np.ones([24,1])*21


    scale=10000
    shapetypes = ['O ring', 'Through hole', 'Blind hole', 'Triangular passage', 'Rectangular passage', 'Circular through slot', 'Triangular through slot', 'Rectangular through slot', 'Rectangular blind slot','Triangular pocket', 'Rectangular pocket', 'Circular end pocket', 'Triangular blind step', 'Circular blind step', 'Rectangular blind step', 'Rectangular through step' , '2-sides through step', 'Slanted through step', 'Chamfer', 'Round', 'Vertical circular end blind slot', 'Horizontal circular end blind slot', '6-sides passage', '6-sides pocket']

    machtypes=np.arange(24)

    '''mesh = pv.read(filename+".stl")
    # Triangulate the mesh
    mesh.triangulate(inplace=True)
    actual=(mesh.volume/1e+12)'''
    vol_pi,calc_info=volcalc_piece(filename,scale)
    vol=volcalc_whole(filename,scale)
    #print(actual,1-(sum(vol_pi)/vol))
        #labels_info = np.array([])
    labels_info = np.empty((0, 5))  # initialize empty 2D array with 4 columns

    for i in labels:
        vol = abs(i[3] - i[0]) * abs(i[4] - i[1]) * abs(i[5] - i[2])
        cor = np.array([(i[3] + i[0]) / 2, (i[4] + i[1]) / 2, (i[5] + i[2]) / 2, vol,i[-2]])
        labels_info = np.vstack((labels_info, cor))  # append new row to 2D array


    # sort rows of labels_info based on last column
    labels_sorted_info = labels_info[np.argsort(labels_info[:, -2])[::-1]]

    calc_sorted_info = calc_info[np.argsort(calc_info[:, -2])[::-1]]

    calc_sorted_info[:, -1]=labels_sorted_info[:, -1]

    # print the sorted array
    #print(labels_sorted_info)
    print(calc_sorted_info)

    cost=np.ones([calc_sorted_info.shape[0],5])*-1

    for i, item in enumerate(calc_sorted_info):
        #Which Operation?
        cost[i][0]=machtypes[int(item[-1])]
        #What is Cutting Speed?
        cost[i][1]=tool[int(item[-1]),1]
        #What is Rate?
        cost[i][2]=rate[int(item[-1])]
        #How many Minutes?
        cost[i][3]=(np.round((item[-2]/1e+08)/tool[int(item[-1]),4],2))+(2.8/calc_info.shape[0])
        #cost[i][3]=np.round((item[-2]/1e+09)/speed[int(item[-1])],2)
        #How much Cost?
        cost[i][4]=np.round((cost[i][2])*cost[i][3],2)
        cost[i][4]=np.round(cost[i][4]+(cost[i][4]*0.1)+(cost[i][4]*0.1)+(cost[i][4]*0.15),2)

    #invoice=grpby(cost)

    df = pd.DataFrame(cost, columns=['Operation','Cutting Speed(in m/min)','Rate(Rs/min)','time','cost'])
    df=df.groupby(['Operation','Cutting Speed(in m/min)','Rate(Rs/min)'])\
        .agg({'time':'sum','cost': 'sum'}).reset_index().sort_values(by='cost', ascending=False)
    df.columns = ['Operation','Cutting Speed','Rate','Time','Cost']
    units=['','','(in m/min)','(Rs/min)','(in min)','(Rs)']
    df=df.reset_index(drop=True)
    df.index = df.index + 1
    df.reset_index(inplace=True)
    
    return df