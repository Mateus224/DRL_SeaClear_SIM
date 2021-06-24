import numpy as np
from pyntcloud import PyntCloud
import csv
import random

class Load_env():
    """"This class loads and preprocess the
    simulated environment"""
    def __init__(self, env_shape):
        self.xn=env_shape[0]
        self.yn=env_shape[1]
        self.zn=env_shape[2]
        self.hashmap = {}
        self.map_2_5D= np.zeros((self.xn, self.yn, 2))


    def store_as_Hash_map(self, voxel):
        """Store the voxelmap in a hashmap
        which can handle up to 1000*1000*1000 voxels
        """
        #from PIL import Image
        for xInd, X in enumerate(voxel):
            for yInd, Y in enumerate(X):
                z_litter=False
                for zInd, Z in enumerate(Y):
                    if (Z == 1 or Z == 0.5):
                        hashkey = 1000000*xInd+1000*yInd+zInd
                        self.hashmap[hashkey] = Z
                        self.map_2_5D[xInd,yInd,0]=zInd
                        if(Z==0.5):
                            z_litter=True
                            self.map_2_5D[xInd,yInd,1]=1
                        elif(Z==1):
                            if(z_litter):
                                self.map_2_5D[xInd,yInd,1]=0
        #self.map_2_5D=self.map_2_5D*4
        #img = Image.fromarray(self.map_2_5D)
        #img.show()




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
        voxel = np.zeros((self.xn, self.yn, self.zn))

        for x, y, z in zip(x_cords, y_cords, z_cords):
            voxel[x][y][z] = 1
            
        minimum= self.search_minimum(voxel)
        voxel= self.shift_to_minimum(voxel, minimum)
        self.store_as_Hash_map(voxel)

        return voxel, self.hashmap

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
                tmp=False
                for zInd, Z in enumerate(Y):
                    if(Z==1):
                        voxel[xInd, yInd, zInd]=0
                        voxel[xInd, yInd, zInd-minimum]=1
                        if(random.randint(0,100)<3):#:(zInd>tmp and xInd==2)
                            tmp=True
                            voxel[xInd][yInd][zInd-minimum] = 0.5
                        if(tmp):
                            voxel[xInd][yInd][zInd-minimum-1] = 0
                            #voxel[xInd][yInd][zInd-minimum-2] = 0
        return voxel

    def readData(self):
        with open('xyz_env/1.xyz') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            next(csv_reader, None)
            data = [data for data in csv_reader]
            data_array = np.asarray(data, dtype=np.float64)
            data_array = np.delete(data_array, 4, axis=1)
            data_array = np.delete(data_array, 3, axis=1)
        return data_array


if __name__ == "__main__":
        pointCloud='xyz_env/1.xyz'
        PCL=readData()
        load_VM(PCL)
