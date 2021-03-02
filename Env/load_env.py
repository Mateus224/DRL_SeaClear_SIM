from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import sys
import csv
import pandas as pd





#def load_VM(pc='xyz_env/1.xyz'):
def load_VM(pc='xyz_env/output.xyz'):

    #cloud=PyntCloud(pd.DataFrame(Cloud, columns=["x","y","z"]))
    cloud = PyntCloud.from_file(pc,
                               sep=" ",
                               header=0,
                              names=["x","y","z"])



    ##print(cloud)
    #cloud.plot(mesh=True, backend="matplotlib")
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=80, n_y=80, n_z=32)
    voxelgrid = cloud.structures[voxelgrid_id]
    #voxelgrid.plot(d=3, mode="density", cmap="hsv")
    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z
    #print(x_cords)
    voxel = np.zeros((80, 80, 32))
    #return x_cords, y_cords, z_cords

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = 1
    minimum=search_minimum(voxel)
    voxel=shift_to_minimum(voxel, minimum)




    ##ax.plot_surface(voxel[x], voxel[y], voxel[z], rstride=1, cstride=1,
                           #linewidth=0, antialiased=False)
    ## cloud.plot(mesh=True, backend="matplotlib")


    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.set_aspect('auto')
    #ax.voxels(voxel, edgecolor="k")
    #plt.show()

    return voxel

def search_minimum(voxel):
    tmp=10000
    for xInd, X in enumerate(voxel):
        for yInd, Y in enumerate(X):
            for zInd, Z in enumerate(Y):
                if(Z==1):
                    if(zInd<tmp):
                        tmp=zInd
    return tmp

def shift_to_minimum(voxel, minimum):
    for xInd, X in enumerate(voxel):
        for yInd, Y in enumerate(X):
            for zInd, Z in enumerate(Y):
                if(Z==1):
                    voxel[xInd, yInd, zInd]=0
                    voxel[xInd, yInd, zInd-minimum]=1
                #if((xInd  == 0) or(yInd  == 0) or(xInd== voxel.shape[0]-1) or(xInd== voxel.shape[1]-1)):
                #    voxel[xInd, yInd, zInd] = 1
    return voxel

def readData():
    with open('xyz_env/1.xyz') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        next(csv_reader, None)
        data = [data for data in csv_reader]
        data_array = np.asarray(data, dtype=np.float64)
        data_array = np.delete(data_array, 4, axis=1)
        data_array = np.delete(data_array, 3, axis=1)

        print(data_array.shape)
    return data_array


if __name__ == "__main__":
    pointCloud='xyz_env/1.xyz'
    PCL=readData()
    load_VM(PCL)
    #load_VM(pointCloud)
