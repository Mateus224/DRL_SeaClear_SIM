import os
import argparse
import render_UUV_env as env3D
import run_terminal as env_terminal


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--networkPath', default='network/', help='folder to put results of experiment in')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    # Environment
    parser.add_argument('--N', type=int, default=25, help='size of grid')
    parser.add_argument('--map_p', type=float, default=.1, help='probability map location is occupied')
    parser.add_argument('--prims', action='store_true', help='prims algorithm for filling in map')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--episode_length', type=int, default=200, help='length of mapping environment episodes')
    # Sensor
    parser.add_argument('--sensor_type', default='local', help='local | range')
    parser.add_argument('--sensor_span', type=int, default=2, help='span of sensor')
    parser.add_argument('--sensor_p', type=float, default=.8, help='probability sensor reading is correct')
    # Visualization for c
    parser.add_argument('--gbp', action='store_false',
                        help='visualize what the network learned with Guided backpropagation')
    parser.add_argument('--gradCAM', action='store_false', help='visualize what the network learned with GradCAM')
    parser.add_argument('--gbp_GradCAM', action='store_false',
                        help='visualize what the network learned with Guided GradCAM')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize what the network learned with Guided GradCAM')
    parser.add_argument('--num_frames', type=int, default=80,
                        help='how many frames have to be stored in the prozessed video')
    parser.add_argument('--beams', type=int, default=20,
                        help='how many beams has the sensor')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def run_render(args):
    #Init render

    env3D.init_render(args)
    return

def run_terminal(args): 
    env_terminal.init(args)
    return
    

def run(args):
    # Initialize sensor
    if args.sensor_type == 'local':
        ism_proto = lambda x: LocalISM(x, span=args.sensor_span, p_correct=args.sensor_p)

    elif args.sensor_type == 'range':
        ism_proto = lambda x: RangeISM(x)
    else:
        raise Exception('sensor type not supported.')

    if args.train_dqn:
        env = MappingEnvironment(ism_proto, N=args.N, p=args.map_p, episode_length=args.episode_length,
                                 prims=args.prims)
        agent = DDDQN_agent(env, args)
        agent.train()

    else:
        env = MappingEnvironment(ism_proto, N=args.N, p=args.map_p, episode_length=args.episode_length,
                                 prims=args.prims)
        agent = DDDQN_agent(env, args)
        if (args.visualize):
            print("<< visualization >>\n")
            visprcess = visprocess(env, args)
            visprcess.visgame(agent)
        else:
            print("<< test dd >>\n")
            rewards = []
            for k in range(1000):
                obs = env.reset()
                env.render(reset=True)
                done = False
                R = 0
                while not done:
                    # Perform a_t according to agent
                    action = agent.make_action(obs)
                    # Receive reward r_t and new state s_t+1
                    obs, reward, done, info = env.step(action)
                    R += reward
                    env.render()
                print(R)
                rewards.append(R)

            np.save(os.path.join(opt.experiment, 'rewards_test'), rewards)




if __name__ == '__main__':
    import numpy as np
    import sys
    import tensorflow as tf

    np.set_printoptions(threshold=sys.maxsize)
    args = parse()
    # make path
    os.makedirs(args.networkPath, exist_ok=True)
    if (False):#not args.do_render):
        print("-----------------------------not rendering-----------------------------\n")
        run_terminal(args)
    else:
        print("reder")
        run_render(args)
