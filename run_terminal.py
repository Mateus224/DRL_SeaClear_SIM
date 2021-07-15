from Env.load_env import Load_env
from Env.env3d_1 import SonarModel, MapEnv
from Agents.DDDQN.DDDQN_agent import DDDQN_agent
import multiprocessing


def init(args, env_shape=[48, 48, 32]):
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
    env=MapEnv(env_shape,pool)
    global obs
    obs, voxelVis, _ = env.reset()
    agent = DDDQN_agent(env, args)
    done=False
    if args.train:
        agent.train()
    else:
        start = time.time()
        while not done:
            action = agent.make_action(obs)
            obs, reward, done, info = env.step(action)
            reward+= reward
        print('total time (s)= ' + str(end-start))
            

def init_base(args, env_shape=[40,40,23]):
    env=Map(env_shape)
    obs, _,_ =env.reset()
    agent = lawn_mower(env,args)
    done = False
    while not done:
        action =agent.make_action(obs)
        obs, reward, done, info = env.step(action)

