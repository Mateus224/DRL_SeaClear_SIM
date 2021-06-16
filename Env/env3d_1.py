from Env.pytransform3d_.transformations import rotate_transform, transform_from, translate_transform
import numpy as np
import random
from Env.load_env import Load_env
from Env.pytransform3d_.rotations import *


def safe_log(x):
    if x <= 0.:
        return 0.
    return np.log(x)
safe_log = np.vectorize(safe_log)

class Pose:
    def __init__(self, x=0, y=0, z=0, orientation=0):
        R= matrix_from_axis_angle([0,0,1,  -np.pi/2 ])# np.pi])
        self.pose_matrix= transform_from(R,[x,y,z])


class SonarModel(object):
    def __init__(self, map, pose, num_beams=15, separation=5.46, beamWidth=82, beamOpening=20, diastance_accuracy=0.985, measure_accurancy=np.array([0.75, 0.85, 0.95, 0.85, 0.85])):
        self.map = map
        self.pose=pose
        self.num_beams=num_beams
        self.separation=(separation*np.pi)/180
        self.beamOpening=(beamOpening*np.pi)/180 # convert into radiant
        self.beamWidth = (beamWidth*np.pi)/180 # convert into radiant
        self.obs_hash_map = {}
        self.update_map ={}
        self.diastance_accuracy=diastance_accuracy
        self.measure_accurancy=measure_accurancy
        self.sensor_matrix=np.zeros((num_beams, int(measure_accurancy.shape[0]), 4,4))
        
        #probability = if 1-e^(x*accuracy)-(1-min_error)<0.5

    def init_sensor(self, render=True):
        self.R_render=self.rotation_matrix_from_vectors([1,1,1], [0,0,-1])
        self.sensor_matrix[:,:,:,3]=self.pose.pose_matrix[:, 3]
        for i in range(self.num_beams):
            for j in range(self.measure_accurancy.shape[0]):
                if .5*self.num_beams < i:
                    if not render:
                        A2B=np.matmul(self.pose.pose_matrix[:3, :3],self.R_render)
                        self.sensor_matrix[i,j,:3, :3] = A2B[:3, :3]
                    else:
                        A2B=np.matmul(self.pose.pose_matrix[:3, :3],self.R_render)
                        self.sensor_matrix[i,j,:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, -(self.num_beams-i)*self.separation]), A2B) 
                        if.5*self.measure_accurancy.shape[0] < j: 
                            self.sensor_matrix[i,j,:3, :3]=np.matmul(matrix_from_axis_angle([0, 1, 0, -(self.measure_accurancy.shape[0]-j)*self.beamOpening/self.measure_accurancy.shape[0]]), self.sensor_matrix[i,j,:3, :3]) 
                        else:
                            self.sensor_matrix[i,j,:3, :3] = np.matmul(matrix_from_axis_angle([0, 1, 0, j*self.separation]), self.sensor_matrix[i,j,:3, :3])

                else:
                    if not render:
                        A2B=np.matmul(self.pose.pose_matrix[:3, :3],self.R_render)
                    else:
                        A2B=np.matmul(self.pose.pose_matrix[:3, :3],self.R_render)
                        self.sensor_matrix[i,j,:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, i*self.separation]), A2B)
                        if.5*self.measure_accurancy.shape[0] < j: 
                            self.sensor_matrix[i,j,:3, :3]=np.matmul(matrix_from_axis_angle([0, 1, 0, -(self.measure_accurancy.shape[0]-j)*self.beamOpening/self.measure_accurancy.shape[0]]), self.sensor_matrix[i,j,:3, :3]) 
                        else:
                            self.sensor_matrix[i,j,:3, :3] = np.matmul(matrix_from_axis_angle([0, 1, 0, j*self.separation]), self.sensor_matrix[i,j,:3, :3])

    def seonsor_collision(self):
        
        #sensor_coordinates = self.pose[:3,:3] * self.sensor_matrix[]

        return self.update_map

    def log_odds(self, pos):
        log_odd_map = np.zeros((self.map.shape[0], self.map.shape[1], self.map.shape[2]))
        return log_odd_map

    def render(self, beams):
        P = np.empty((len(self.sensor_matrix[0])*len(self.sensor_matrix[1]), 3))
        A2C=np.eye(4)
        A2C[:3, 3] = self.pose.pose_matrix[:3, 3]
        for d in range(P.shape[1]):
            P[:, d] = np.linspace(0, 10, len(P))
        for i in range(self.sensor_matrix.shape[0]):
            for j in range(self.sensor_matrix.shape[1]):
                A2C[:3, :3] =self.sensor_matrix[i][j][:3, :3]
                beams[self.sensor_matrix.shape[1]*i+j].set_data(P, A2C.copy())
        return beams


    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix


        ##get the Beam points
class MapEnv(object):
    def __init__(self, env_shape, p=.1, episode_length=1000, randompose=True):
        self.random_pose=False
        self.p = p
        self.env_shape=env_shape
        self.xn = env_shape[0]
        self.yn = env_shape[1]
        self.zn = env_shape[2]
        self.ACTIONS = np.array([[0.5, 0, 0, 0, 0, 0],
                   [-0.5, 0, 0, 0, 0, 0],
                   [0, 0.5, 0,  0, 0, 0],
                   [0, -0.5, 0, 0, 0, 0],
                   [0, 0, 0.5, 0, 0, 0],
                   [0, 0, -0.5, 0, 0, 0],
                   [0, 0, 0, 0.2356194, 0, 0 ],
                   [0, 0, 0, -0.2356194, 0, 0 ],
                   [0, 0, 0, 0, 0.2356194, 0 ],
                   [0, 0, 0, 0, -0.2356194, 0],
                   [0, 0, 0, 0, 0, 0.2356194 ],
                   [0, 0, 0, 0, 0, -0.2356194 ]])
           

        self.episode_length = episode_length
        self.prims = False

        self.pq = self.init_pose(x=0,y=0,z=0) # position / quaternion
        self.pose=np.array([self.xn/2,self.yn/2,self.zn/2,self.pq[0],self.pq[1],self.pq[2],self.pq[3]])
        self.t = None
        self.viewer = None
        self.ent = None

    def init_pose(self,x,y,z):
        xyz=[x,y,z]
        rotation_m=matrix_from_euler_xyz(xyz)
        pq=quaternion_from_matrix(rotation_m)
        return pq





    def reset(self, num_beams=15):
        # generate new map
        random_map=np.random.randint(0, 2)
        map_const=Load_env(self.env_shape)
        self.map, self.hashmap = map_const.load_VM()
        self.has_map=map_const.hashmap

        # generate initial pose
        if self.random_pose:
            self.x0, self.y0, self.z0= np.random.randint(0, self.xn), np.random.randint(0, self.yn), np.random.randint(3, self.zn)
        else:
            self.x0, self.y0, self.z0 = self.xn/2, self.yn/2, self.zn/2
        self.pose = Pose(self.x0, self.y0, self.z0)
        #sself.map[self.x0, self.y0] = 0

        # reset inverse sensor model, likelihood and pose
        self.sonar_model = SonarModel(self.map, self.pose, num_beams=num_beams)
        self.sonar_model.init_sensor()
        self.l_t = np.zeros((self.xn, self.yn, self.zn))
        self.t = 0

        return self.get_observation() , self.map, self.sonar_model.sensor_matrix



    def in_map(self, new_pos):
        x = new_pos[0]
        y = new_pos[1]
        z = new_pos[2]
        # TODO: Implement ground ristrictions
        return x >= 0 and y >= 0 and x < self.xn and y < self.yn and z >= 0 and z < self.zn
    
    def collision(self,new_pos):
        x = int(new_pos[0])
        y = int(new_pos[1])
        z = int(new_pos[2])
        hashkey = 1000000*x+1000*y+z
        if hashkey in self.hashmap:
            print("COLLISION ! ! !")
            return True
        else:
            return False

    def legal_change_in_pose(self,new_position):
        if self.in_map(new_position) and not self.collision(new_position):
            return True
        else:
            return False
        #return self.in_map(self.pose[0] + change_pose[0], pose[1] + change_pose[1], pose[2] + change_pose[2]) and self.map[pose[0] + change_pose[0], pose[1] + change_pose[1], pose[2] + change_pose[2]] == 0
        #self.in_map(self.pose[:3] + change_pose[:3]) #and self.map[pose[0] + change_pose[0], pose[1] + change_pose[1], pose[2] + change_pose[2]] == 0

    def logodds_to_prob(self, l_t):
        return 1 - 1. / (1 + np.exp(l_t))

    def calc_entropy(self, l_t):
        p_t = self.logodds_to_prob(l_t)
        entropy = - (p_t * safe_log(p_t) + (1 - p_t) * safe_log(1 - p_t))

        return entropy

    def calc_sum_entropy(self, obs):
        np.sum(obs)
        return

    def observation_size(self):
        return 2 * self.N - 1

    def get_observation(self):
        #sensor_collision()
        #augmented_p = float("inf") * np.ones((self.env_shape[0], self.env_shape[1] , self.env_shape[2]))
        #augmented_p[self.N - 1:2 * self.N - 1, self.N - 1:2 * self.N - 1] = self.l_t
        obs=self.l_t
        #obs = augmented_p[self.pose.x:self.pose.x + 2 * self.N - 1, self.pose.y:self.pose.y + 2 * self.N - 1]
        p = self.logodds_to_prob(obs)
        self.ent = self.calc_entropy(obs)
        ent = self.ent

        ### scale p to [-1, 1]
        p = (p - .5) * 2

        ### scale entropy to [-1, 1]
        ent /= -np.log(.5)

        ent = (ent - .5) * 2

        return np.concatenate([np.expand_dims(p, -1), np.expand_dims(ent, -1)], axis=-1)
        

    def step(self, a):
        # Step time
        if self.t is None:
            print("Must call env.reset() before calling step()")
            return
        self.t += 1
        a=1
        # Perform action
        
        R_t=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        if a<6:
            new_position=self.pose.pose_matrix[:3,3] + self.ACTIONS[a][:3]
            if self.legal_change_in_pose(new_position):
                self.pose.pose_matrix[:3,3]= new_position

        elif a==6 or a==7:
            R_t=matrix_from_axis_angle([1, 0, 0,self.ACTIONS[a][3]])
            self.pose.pose_matrix[:3,:3]= np.matmul(self.pose.pose_matrix[:3,:3],R_t)

        elif a==8 or a==9:
            R_t=matrix_from_axis_angle([0, 1, 0,self.ACTIONS[a][4]])
            self.pose.pose_matrix[:3,:3]= np.matmul(self.pose.pose_matrix[:3,:3],R_t)
        
        elif a==10 or a==11:
            R_t=matrix_from_axis_angle([0, 0, 1,self.ACTIONS[a][5]])
            self.pose.pose_matrix[:3,:3]= np.matmul(self.pose.pose_matrix[:3,:3],R_t)


        # bayes filter
        new_l_t = self.l_t + self.sonar_model.log_odds(self.pose)
        # reward is decrease in entropy
        reward = np.sum(self.calc_entropy(self.l_t)) - np.sum(self.calc_entropy(new_l_t))

        # Check if done
        done = False
        if self.t == self.episode_length:
            done = True
            self.t = None

        self.l_t = new_l_t

        return self.get_observation(), reward, done, self.pose.pose_matrix