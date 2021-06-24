import Env.pytransform3d_.visualizer as pv
from Env.pytransform3d_.rotations import *
from Env.pytransform3d_.transformations import *
from Env.pytransform3d_.batch_rotations import *
import open3d as o3d
from Env.pytransform3d_.plot_utils import plot_box
#import Env.load_env as env
from Env.load_env import Load_env
from Env.env3d_1 import SonarModel, MapEnv
from Agents.DDDQN.DDDQN_agent import DDDQN_agent
import xxhash

import sys
import os

obs = None



hash_box_map={}
box=None

def update_map(fig, step, belief, update_map):
    global box
    for hashkey in update_map:
        
        x= int(hashkey / 1000000)
        y= int((hashkey - (x*1000000))/1000 )
        z= int(hashkey-(x*1000000)-(y*1000))
        value=belief[x,y,z]
        box=hash_box_map.get(hashkey)
        box.remove_artist2(fig)
        vox = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        box = pv.Box(size=[1,1,1],A2B=vox, c=[1-value,1-value,0.5])
        box.add_artist2(fig)
        hash_box_map[hashkey] = box


    return box

def animation_callback1(step, n_frames, frame, frame_debug, uuv, beams, env, agent, fig):
    global obs
    action = agent.make_action(obs)
    if step % 4:
        action=8
        if step % 5:
            action=8
    else:
        action=1


    obs, reward, done, uuv_pose = env.step(action)
    belief=env.prob
    update_map_=env.update_map
    beams=env.sonar_model.render(beams)
    box=update_map(fig, step, belief, update_map_)
    #animate_sensor(15, step, sensor_matrix)


    reward+= reward
    A2B = np.eye(4)
    A2B[0, 3] = 6+step

    uuv.set_data(A2B)
    uuv.set_data(uuv_pose)

    return uuv, beams, box#frame, frame_debug, uuv, beams



def animation_callback(step, n_frames, frame, frame_debug, uuv, beams, Env, Agent):
    angle = 2.0 * np.pi * (step + 1) / n_frames
    Base_uuv_q = quaternion_from_axis_angle([0,0,1, -3 * np.pi / 2])
    R3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    R_q=quaternion_from_axis_angle([0,1,0, angle])
    new_q= concatenate_quaternions(Base_uuv_q, R_q)

    #R2=rotation_matrix_from_vectors([1,1,1], [0,0,-1])

    R1 =matrix_from_quaternion(new_q)
    R2=np.matmul(R1,R1)#R2)
    R = matrix_from_angle(0, angle)
    A2B = np.eye(4)
    A2B[:3, :3] = R
    A2B[2,3]=6
    A2B[1, 3] = 20
    A2B[0, 3] = 6+step
    point=[5+(step/2), 34, 20]
    point2 = [24, -0.49, 8]
    A2UUV = transform_from(R1, point)
    Sensor =transform_from(R2, point)
    Frame = transform_from(R1, point)
    Frame_debug=transform_from(R3, point2)
    frame_debug.set_data(Frame_debug)
    frame.set_data(Frame)
    print(A2UUV, "A2UUV")

    uuv.set_data(A2UUV)
    animate_sensor(beams, step, Sensor)
    print(beams[1].get_line_coordinate(0)[1][1])
    print(beams[1].get_line_coordinate(1))
    print(beams[1].get_line_coordinate(2))
    print(beams[1].get_line_coordinate(3))
    print(beams[1].get_line_coordinate(18))
    print(beams[1].get_line_coordinate(19))
    print(beams[1].get_line_coordinate(20))
    print(beams[1].get_line_coordinate(21))
    return frame, frame_debug, uuv, beams




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

def animate_sensor(beams, sensor_matrix):
    P = np.empty((beams, 3))
    A2C=np.eye(4)
    A2C[:, 3] = A2B[:, 3]
    for d in range(P.shape[1]):
        P[:, d] = np.linspace(0, 10, len(P))
    for i in range(len(beams)):
        if .5*len(beams) < i:
            A2C[:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, -i*0.0055555]), A2B[:3, :3])
        else:
            A2C[:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, (i-.5*len(beams)) * 0.0055555]), A2B[:3, :3])
        beams[i].set_data(P, A2C.copy())
    return beams

def animate_sensor1(beams,  step, A2B):
    t = step*10
    P = np.empty((20, 3))
    A2C=np.eye(4)
    A2C[:, 3] = A2B[:, 3]
    for d in range(P.shape[1]):
        P[:, d] = np.linspace(0, 10, len(P))
    for i in range(len(beams)):
        if .5*len(beams) < i:
            A2C[:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, -i*0.0055555]), A2B[:3, :3])
        else:
            A2C[:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, (i-.5*len(beams)) * 0.0055555]), A2B[:3, :3])
        beams[i].set_data(P, A2C.copy())
    return beams







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
