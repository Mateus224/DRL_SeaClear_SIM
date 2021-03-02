import pytransform3d_.visualizer as pv
from pytransform3d_.rotations import *
from pytransform3d_.transformations import *
from pytransform3d_.batch_rotations import *
import open3d as o3d
from pytransform3d.plot_utils import plot_box
import load_env as env
import sys
import os


def rotation_matrix_from_vectors(vec1, vec2):
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


def animation_callback(step, n_frames, frame, frame_debug, uuv, beams):
    angle = 2.0 * np.pi * (step + 1) / n_frames
    Base_uuv_q = quaternion_from_axis_angle([0,0,1, -3 * np.pi / 2])
    R3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    R_q=quaternion_from_axis_angle([0,1,0, angle])
    new_q= concatenate_quaternions(Base_uuv_q, R_q)


    R2=rotation_matrix_from_vectors([1,1,1], [0,0,-1])


    R1 =matrix_from_quaternion(new_q)
    R2=np.matmul(R1,R2)

    R = matrix_from_angle(0, angle)
    A2B = np.eye(4)
    A2B[:3, :3] = R
    A2B[2,3]=6
    A2B[1, 3] = 20
    A2B[0, 3] = 6+step
    point=[5+(step/10), 20, 6]
    point2 = [24, -0.49, 8]
    A2UUV = transform_from(R1, point)
    Sensor =transform_from(R2, point)
    Frame = transform_from(R1, point)
    Frame_debug=transform_from(R3, point2)
    frame_debug.set_data(Frame_debug)
    frame.set_data(Frame)

    uuv.set_data(A2UUV)
    animate_sensor(beams, step, Sensor)
    return frame, frame_debug, uuv, beams




def init_animate_sensor(fig, beams_i):
    R=matrix_from_axis_angle([0,0,1, -3 * np.pi / 2])
    R=[[1,0,0],[0,1,0],[0,0,1]]
    P = np.zeros((20, 3))
    colors = np.empty((19, 3))
    for d in range(colors.shape[1]):
        P[:, d] = np.linspace(0, 10, len(P))
        colors[:, d] = np.linspace(0, 1, len(colors))

    eye=np.eye(4)
    eye[:3,:3]=R
    lines = list()
    for x in range(beams_i):
        lines.append(fig.plot(P, eye, colors))

    fig.view_init()

    return fig, lines

def animate_sensor(beams,  step, A2B):
    t = step*10
    P = np.empty((20, 3))
    A2C=np.eye(4)
    A2C[:, 3] = A2B[:, 3]
    for d in range(P.shape[1]):
        P[:, d] = np.linspace(0, 10, len(P))
    for i in range(len(beams)):
        if(.5*len(beams)<i):
            A2C[:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, -i*0.0055555]), A2B[:3, :3])
        else:
            A2C[:3, :3] = np.matmul(matrix_from_axis_angle([1, 0, 0, (i-.5*len(beams)) * 0.0055555]), A2B[:3, :3])
        beams[i].set_data(P, A2C.copy())
    return beams



def init_env():
    BASE_DIR = "../Mashes/"
    data_dir = BASE_DIR
    search_path = "."
    while (not os.path.exists(data_dir) and
           os.path.dirname(search_path) != "pytransform3d"):
        search_path = os.path.join(search_path, "..")
        data_dir = os.path.join(search_path, BASE_DIR)
    fig = pv.figure(width=500, height=500)
    frame = fig.plot_basis(R=np.eye(3), s=2)
    frame_debug = fig.plot_basis(R=np.eye(3), s=2)
    R = matrix_from_angle(2,3*np.pi/2)
    A2C = np.eye(4)
    A2C[:3, :3] = R
    uuv = pv.Mesh("../Mashes/uuv.stl",s=[0.3,0.225,0.3], c=[0.4,0.3,0.2])
    uuv.add_artist2(fig)
    uuv.add_artist(fig)

    return fig, uuv, frame, frame_debug


def build_env(fig):

    voxel = env.load_VM()
    for xInd, X in enumerate(voxel):
        for yInd, Y in enumerate(X):
            for zInd, Z in enumerate(Y):
                if(Z==1):
                    vox = np.array([[1, 0, 0, xInd], [0, 1, 0, yInd], [0, 0, 1, zInd], [0, 0, 0, 1]])
                    box = pv.Box(size=[1,1,1],A2B=vox, c=[1,0.89,0.707])
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




if "__file__" in globals():

    fig, uuv, frame, frame_debug =init_env()
    beams_i=20
    fig = build_env(fig)
    fig, beams = init_animate_sensor(fig, beams_i)
    n_frames = 100
    fig.animate(animation_callback, n_frames, fargs=(n_frames, frame, frame_debug, uuv, beams), loop=True)

    fig.show()
