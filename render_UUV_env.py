import Env.pytransform3d_.visualizer as pv
from Env.pytransform3d_.rotations import *
from Env.pytransform3d_.transformations import *
from Env.pytransform3d_.batch_rotations import *
import open3d as o3d
from Env.pytransform3d_.plot_utils import plot_box

from Env.load_env import Load_env
from Env.env3d_1 import SonarModel, MapEnv
from Agents.DDDQN.DDDQN_agent import DDDQN_agent
import xxhash
from random import randrange
import sys
import os
import copy

obs = None



hash_box_map={}
def update_map(fig, step, belief, update_map):
    for hashkey in update_map:
        box=hash_box_map.get(hashkey)
        #if step==0:
        #    box2=copy.deepcopy(box)
        
        
        x= int(hashkey / 1000000)
        y= int((hashkey - (x*1000000))/1000 )
        z= int(hashkey-(x*1000000)-(y*1000))
        value=belief[x,y,z]
        box.update_color2(fig,[1-value,1-value,0.5])
        #if step==0:
        #    box.remove_artist(fig)
            #vox = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
            #box = pv.Box(size=[1,1,1],A2B=vox, c=[1,0.89,0.707])
        #    box2.add_artist(fig)

        



def animation_callback1(step, n_frames, frame, frame_debug, uuv, beams, env, agent, fig):
    global obs
    

    #action=randrange(12)
    action = agent.make_action(obs)

    action=randrange(12)
    action = agent.make_action(obs)
    if step % 3==0:
        action=7
    elif step %3==1:
        action =8
    else:
        action =10
    action=randrange(1,12)
    #action=11
    #print(action)
    obs, reward, done, uuv_pose = env.step(action)
    
    belief=env.prob
    update_map_=env.update_map
    beams=env.sonar_model.render(beams)
    uuv.set_data(uuv_pose)
    update_map(fig, step, belief, update_map_)


    reward+= reward


    return uuv, beams #frame, frame_debug, uuv, beams




def init_animate_sensor(fig, beams_i):
    R=[[1,0,0],[0,1,0],[0,0,1]]
    P = np.zeros((20, 3))
    colors = np.empty((19, 3))
    for d in range(colors.shape[1]):
        P[:, d] = np.linspace(0, 10, len(P))
        colors[:, d] = np.linspace(0, 1, len(colors))

    eye=np.eye(4)
    eye[:3,:3]=R
    lines = list()
    for _ in range(beams_i*5):
        lines.append(fig.plot(P, eye, colors))

    fig.view_init()

    return fig, lines



def build_env(fig, env):
    global boxes
    #voxel = env.load_VM()
    voxel=env
    for xInd, X in enumerate(voxel):
        for yInd, Y in enumerate(X):
            for zInd, Z in enumerate(Y):
                if(Z==1 or Z==0.5):
                    hashkey = 1000000*xInd+1000*yInd+zInd
                    vox = np.array([[1, 0, 0, xInd], [0, 1, 0, yInd], [0, 0, 1, zInd], [0, 0, 0, 1]])
                    box = pv.Box(size=[1,1,1],A2B=vox, c=[1,0.89,0.707])
                    box.add_artist2(fig)
                    hash_box_map[hashkey] = box
                    if Z==0.5:
                        box2 = pv.Box(size=[1,1,1],A2B=vox, c=[.5,0.5,.5])
                        box2.add_artist(fig)
                    else:
                        box.add_artist(fig)
                if((xInd  == 0) and (yInd  == 0)):
                    wall = np.array([[1, 0, 0, -1], [0, 1, 0,( voxel.shape[1]/2)+.5], [0, 0, 1, (voxel.shape[2]/2)+.5], [0, 0, 0, 1]])
                    box = pv.Box(size=[1, voxel.shape[1], voxel.shape[2]], A2B=wall)
                    box.add_artist2(fig)
                    box.add_artist(fig)

                    wall = np.array([ [1, 0, 0, (voxel.shape[1] / 2) +.5], [0, 1, 0, -1],[0, 0, 1, (voxel.shape[2] / 2) + .5],[0, 0, 0, 1]])
                    box = pv.Box(size=[ voxel.shape[0], 1,voxel.shape[2]], A2B=wall)
                    box.add_artist(fig)
                    box.add_artist2(fig)
                if ((xInd == 0) and (yInd == 0)and (zInd == 0)):
                    wall = np.array(
                        [[1, 0, 0, (voxel.shape[1] / 2) +.5], [0, 1, 0, (voxel.shape[1] / 2) +.5], [0, 0, 1, -1],
                         [0, 0, 0, 1]])
                    box = pv.Box(size=[voxel.shape[0], voxel.shape[1], 1], A2B=wall)
                    box.add_artist(fig)
                    box.add_artist2(fig)

    return fig

def init_env():

    BASE_DIR = "Mashes/"
    data_dir = BASE_DIR
    search_path = "."
    while (not os.path.exists(data_dir) and
           os.path.dirname(search_path) != "pytransform3d_"):
        search_path = os.path.join(search_path, "..")
        data_dir = os.path.join(search_path, BASE_DIR)
    fig = pv.figure(width=500, height=500)
    frame = fig.plot_basis(R=np.eye(3), s=2)
    frame_debug = fig.plot_basis(R=np.eye(3), s=2)
    R = matrix_from_angle(2,3*np.pi/2)
    A2C = np.eye(4)
    A2C[:3, :3] = R
    uuv = pv.Mesh("Mashes/uuv.stl",s=[0.3,0.225,0.3], c=[0.4,0.3,0.2])
    uuv.add_artist2(fig)
    uuv.add_artist(fig)

    return fig, uuv, frame, frame_debug

def init_render(args, env_shape=[40, 40, 23], num_beams=15):
    Env=MapEnv(env_shape)
    global obs
    obs, voxelVis, sensor_matrix = Env.reset(num_beams)
    fig, uuv, frame, frame_debug = init_env()
    fig = build_env(fig, voxelVis)
    fig, beams = init_animate_sensor(fig, num_beams)
    n_frames = 100


    Agent = DDDQN_agent(Env, args)
    fig.animate(animation_callback1, n_frames, fargs=(n_frames, frame, frame_debug, uuv, beams, Env, Agent, fig), loop=True)

    fig.show()
