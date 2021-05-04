import numpy as np
import random
from Env.load_env import Load_env
from Env.pytransform3d_.rotations import *

class Pose:
    def __init__(self, x=0, y=0, z=0, orientation=0):
        self.x = x
        self.y = y
        self.z = z
        self.orientation = orientation

class SonarModel(object):
    def __init__(self, map, beams=20, beamWidth=1, min_error=0.05, accuracy=0.01):
        self.map = map
        self.beams = beams                     # how many beams are simulated
        self.beamWidth = (beamWidth*np.pi)/180 # convert into radiant
        self.min_error = min_error
        self.accuracy = accuracy
        #probability = if 1-e^(x*accuracy)-(1-min_error)<0.5

    def log_odds(self, pos):
        log_odd_map = np.zeros((self.map[0], self.map[1], self.map[2]))

        return log_odd_map

        ##get the Beam points
class MapEnv(object):
    def __init__(self, xn=70, yn=70, zn=30, p=.1, episode_length=1000, randompose=True):
        self.random_pose=True
        self.p = p
        self.xn=xn
        self.yn = yn
        self.zn = zn


        self.episode_length = episode_length
        self.prims = False

        self.pq = self.init_pose(x=0,y=0,z=0) # position / quaternion
        self.pose=np.array([xn,yn,zn,self.pq[0],self.pq[1],self.pq[2],self.pq[3]])
        print(self.pose)
        self.t = None
        self.viewer = None
        self.ent = None

    def init_pose(self,x,y,z):
        xyz=[x,y,z]
        rotation_m=matrix_from_euler_xyz(xyz)
        pq=quaternion_from_matrix(rotation_m)
        return pq


    def colision_detection(self):

        return


    def reset(self):
        # generate new map
        random_map=np.random.randint(0, 2)
        map_const=Load_env(self.xn,self.yn,self.zn)
        self.map=map_const.load_VM()
        self.has_map=map_const.hashmap

        # generate initial pose
        if self.random_pose:
            self.x0, self.y0, self.z0= np.random.randint(0, self.xn), np.random.randint(0, self.yn), np.random.randint(20, self.zn)
        else:
            self.x0, self.y0, self.z0 = self.xn/2, self.yn/2, self.zn/2
        #self.pose = Pose(self.x0, self.y0, 0)
        #sself.map[self.x0, self.y0] = 0

        # reset inverse sensor model, likelihood and pose
        self.ism = SonarModel(self.map)
        self.l_t = np.zeros((self.xn, self.yn, self.zn))
        self.t = 0

        return self.get_observation() , self.map



    def in_map(self, x, y, z):
        return x >= 0 and y >= 0 and x < self.N and y < self.N

    def legal_change_in_pose(self, pose, dx, dy):
        return self.in_map(pose.x + dx, pose.y + dy) and self.map[pose.x + dx, pose.y + dy] == 0

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
        #augmented_p = float("inf") * np.ones((3 * self.N - 2, 3 * self.N - 2))
        #augmented_p[self.N - 1:2 * self.N - 1, self.N - 1:2 * self.N - 1] = self.l_t
        #obs = augmented_p[self.pose.x:self.pose.x + 2 * self.N - 1, self.pose.y:self.pose.y + 2 * self.N - 1]
        #p = self.logodds_to_prob(obs)
        #self.ent = self.calc_entropy(obs)
        #ent = self.ent

        ### scale p to [-1, 1]
        #p = (p - .5) * 2

        ### scale entropy to [-1, 1]
        #ent /= -np.log(.5)

        #ent = (ent - .5) * 2
        #return np.concatenate([np.expand_dims(p, -1), np.expand_dims(ent, -1)], axis=-1)
        return

    def step(self, a):
        # Step time
        if self.t is None:
            print("Must call env.reset() before calling step()")
            return
        self.t += 1

        # Perform action
        dx, dy, dr = self.ACTIONS[a]
        if self.legal_change_in_pose(self.pose, dx, dy):
            self.pose.x += dx
            self.pose.y += dy
            self.pose.orientation = (self.pose.orientation + dr) % 360

        # bayes filter
        new_l_t = self.l_t + self.ism.log_odds(self.pose)
        print(new_l_t.shape)
        # reward is decrease in entropy
        reward = np.sum(self.calc_entropy(self.l_t)) - np.sum(self.calc_entropy(new_l_t))

        # Check if done
        done = False
        if self.t == self.episode_length:
            done = True
            self.t = None

        self.l_t = new_l_t

        return self.get_observation(), reward, done, None