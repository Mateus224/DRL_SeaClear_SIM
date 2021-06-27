from Env.load_env import Load_env
from Env.env3d_1 import SonarModel, MapEnv
from Agents.DDDQN.DDDQN_agent import DDDQN_agent


def init(args, env_shape=[40, 40, 23]):
    env=MapEnv(env_shape)
    global obs
    obs, voxelVis, _ = env.reset()
    agent = DDDQN_agent(env, args)
    done=False
    if args.train:
        agent.train()
    else:
        while not done:
            action = agent.make_action(obs)
            obs, reward, done, info = env.step(action)
            reward+= reward