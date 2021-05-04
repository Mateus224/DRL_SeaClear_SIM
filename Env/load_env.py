import numpy as np
from pyntcloud import PyntCloud
import csv

class Load_env():
    """"This class loads and preprocess the
    simulated environment"""
    def __init__(self, xn, yn ,zn):
        self.xn=xn
        self.yn=yn
        self.zn=zn
        self.hashmap = {}

    def store_as_Hash_map(self, voxel):
        """Store the voxelmap in a hashmap
        which can handle up to 1000*1000*1000 voxels
        """

        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                for zInd, Z in enumerate(Y):
                    if Z == 1:
                        hashkey = 1000000*xInd+1000*yInd+zInd
                        self.hashmap[hashkey] = 1



    def load_VM(self, pc='Env/xyz_env/point_cloud.xyz'):
        #random_map=1
        #cloud=PyntCloud(pd.DataFrame(Cloud, columns=["x","y","z"]))
        cloud = PyntCloud.from_file(pc,
                                   sep=" ",
                                   header=0,
                                  names=["x","y","z"])


        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=self.xn, n_y=self.yn, n_z=self.zn)
        voxelgrid = cloud.structures[voxelgrid_id]
        x_cords = voxelgrid.voxel_x
        y_cords = voxelgrid.voxel_y
        z_cords = voxelgrid.voxel_z
        voxel = np.zeros((80, 80, 32))

        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = 1
        minimum= self.search_minimum(voxel)
        voxel= self.shift_to_minimum(voxel, minimum)
        self.store_as_Hash_map(voxel)

        return voxel

    def search_minimum(self, voxel):
        tmp=10000
        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                for zInd, Z in enumerate(Y):
                    if(Z==1):
                        if(zInd<tmp):
                            tmp=zInd
        return tmp

    def shift_to_minimum(self, voxel, minimum):
        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                for zInd, Z in enumerate(Y):
                    if(Z==1):
                        voxel[xInd, yInd, zInd]=0
                        voxel[xInd, yInd, zInd-minimum]=1
                    #if((xInd  == 0) or(yInd  == 0) or(xInd== voxel.shape[0]-1) or(xInd== voxel.shape[1]-1)):
                    #    voxel[xInd, yInd, zInd] = 1
        return voxel

    def readData(self):
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
