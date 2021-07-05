from ..agent import Agent
import os

import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Multiply, Lambda
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Concatenate
from tensorflow.keras.utils import plot_model

#from tensorflow.compat.v1.keras.backend import set_session

# random#.seed(1)
# np.random#.seed(1)
#tf.reset_default_graph()


# tf.set_random_seed(1)

# reference : https://github.com/tokb23/dqn/blob/master/dqn.py

class DDDQN_agent(Agent):

    def __init__(self, env, args):
        import numpy as np
        from tensorflow.keras import backend as K
        import sys




        # parameters
        self.frame_x = env.xn
        self.frame_y = env.yn
        #self.frame_z = env.zn
        self.pose=env.pose
        self.num_steps = args.num_steps
        self.state_length = 1
        self.gamma = args.gamma
        self.exploration_steps = args.exploration_steps
        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.initial_replay_size = args.initial_replay_size
        self.num_replay_memory = args.num_replay_memory
        self.batch_size = args.batch_size
        self.target_update_interval = args.target_update_interval
        self.train_interval = args.train_interval
        self.learning_rate = args.learning_rate
        self.min_grad = args.min_grad
        self.save_interval = args.save_interval
        self.no_op_steps = args.no_op_steps
        self.save_network_path = args.save_network_path
        self.save_summary_path = args.save_summary_path
        self.test_dqn_model_path = args.test_dqn_model_path
        self.exp_name = args.exp_name
        self.ddqn = True# args.ddqn
        self.dueling = args.dueling


        if args.optimizer.lower() == 'adam':
            self.opt = Adam(lr=self.learning_rate)
        else:
            self.opt = RMSprop(lr=self.learning_rate, decay=0, rho=0.99, epsilon=self.min_grad)

        # environment setting

        self.env = env
        self.num_actions = 12

        self.epsilon = self.initial_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.t = 0

        # Input that is not used when fowarding for Q-value
        # or loss calculation on first output of model
        self.dummy_input = np.zeros((1, self.num_actions))
        self.dummy_batch = np.zeros((self.batch_size, self.num_actions))

        # for summary & checkpoint
        self.total_reward = 0.0
        self.total_q_max = 0.0
        self.total_loss = 0.0
        self.duration = 0
        self.episode = 0
        self.last_30_reward = deque()
        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)
        if not os.path.exists(self.save_summary_path):
            os.makedirs(self.save_summary_path)

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.q_network = self.build_network()

        # Create target network
        self.target_network = self.build_network()

        # load model for testing, train a new one otherwise
        if args.test_dqn:
            self.q_network.load_weights(self.test_dqn_model_path)
            self.log = open(self.save_summary_path + self.exp_name + '.log', 'w')

        else:
            self.log = open(self.save_summary_path + self.exp_name + '.log', 'w')

        # Set target_network weight
        self.target_network.set_weights(self.q_network.get_weights())

    def init_game_setting(self):
        pass

    def train(self):
        while self.t <= self.num_steps:
            terminal = False
            observation,_,_ = self.env.reset()
            while not terminal:
                last_observation = observation
                action = self.make_action(last_observation, test=False)
                observation, reward, terminal, _ = self.env.step(action)
                self.run(last_observation, action, reward, terminal, observation)

    def make_action(self, observation, test=True):
        """
        ***Add random action to avoid the testing model stucks under certain situation***
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        if not test:
            if self.epsilon >= random.random() or self.t < self.initial_replay_size:
                action = random.randrange(self.num_actions)
            else:
                action =np.argmax(self.q_network.predict([np.expand_dims(observation[0], axis=0), np.expand_dims(observation[1][:3], axis=0),np.expand_dims(observation[1][3:], axis=0),  self.dummy_input])[0])
 
            # Anneal epsilon linearly over time
            
            if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:

                if self.epsilon < 0.2:
                    self.epsilon-= self.epsilon_step/2
                elif self.epsilon < 0.1:
                    self.epsilon-= self.epsilon_step/4
                elif self.epsilon < 0.05:
                    self.epsilon-= self.epsilon_step/8
                else:
                    self.epsilon -= self.epsilon_step
        else:
            if 0.005 >= random.random():
                action = random.randrange(self.num_actions)
            else:
                #print(observation.shape)

                action = np.argmax(self.q_network.predict([np.expand_dims(observation[0], axis=0), np.expand_dims(observation[1][:3], axis=0),np.expand_dims(observation[1][3:], axis=0),  self.dummy_input])[0])
 
        return action

    def build_network(self):
        input_position= Input(shape=(3))
        input_pose= Input(shape=(4))
        hidden_feature_pos = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_pose)
        hidden_feature_pose1 = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_feature_pos)

        hidden_feature_position = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_position)
        hidden_feature_position = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(hidden_feature_position)
        comb_pose = Concatenate()([hidden_feature_pose1, hidden_feature_position])

        comb_pose1 = Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(comb_pose)
        comb_pose2 = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(comb_pose1)


        input_frame = Input(shape=(self.frame_x, self.frame_y,3))
        action_one_hot = Input(shape=(self.num_actions,))
        conv1 = Conv2D(64, (4, 4), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_frame)
        conv2 = Conv2D(128, (4, 4), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(conv1)
        conv3 = Conv2D(256, (4, 4), strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.01))(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(flat_feature)
        combine= Concatenate()([hidden_feature, comb_pose2])
        hidden_feature_comb=Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(combine)


        if True:#self.dueling:
            value_hidden = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name = 'value_fc')(hidden_feature_comb)
            value = Dense(1, name = "value")(value_hidden)
            action_hidden = Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name = 'action_fc')(hidden_feature_comb)
            action = Dense(self.num_actions, name = "action")(action_hidden)
            action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keepdims = True), name = 'action_mean')(action) 
            q_value_prediction = Lambda(lambda x: x[0] + x[1] - x[2], name = 'duel_output')([action, value, action_mean])
        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        #select_q_value_of_action = merge([q_value_prediction, action_one_hot], mode='mul',
        #                                 output_shape=(self.num_actions,))

        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=lambda_out_shape)(
            select_q_value_of_action)

        model = Model(inputs=[input_frame, input_position, input_pose, action_one_hot], outputs=[q_value_prediction, target_q_value])

        # MSE loss on target_q_value only
        model.compile(loss=['mse', 'mse'], loss_weights=[0.0, 1.0], optimizer=Adam(lr=0.00001))  # self.opt)
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def build_network_2_D(self):
        # Consturct model
        input_voxel_frame = Input(shape=(self.frame_x, self.frame_y,self.frame_z, 2))
        action_one_hot = Input(shape=(self.num_actions,))
        conv1 = Conv3D(32, kernel_size=(6, 6, 6),  activation='relu')(input_voxel_frame)
        conv2 = Conv3D(64, (4, 4, 4),  activation='relu')(conv1)
        conv3 = Conv3D(64, (2, 2, 2), activation='relu')(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512)(flat_feature)
        lrelu_feature = LeakyReLU()(hidden_feature)
        q_value_prediction = Dense(self.num_actions)(lrelu_feature)

        if True:#self.dueling:
            # Dueling Network
            # Q = Value of state + (Value of Action - Mean of all action value)
            #hidden_feature_2 = Dense(512, activation='relu')(flat_feature)
            #state_value_prediction = Dense(1)(hidden_feature_2)
            #q_value_prediction = merge([q_value_prediction, state_value_prediction],
            #                           mode=lambda x: x[0] - K.mean(x[0]) + x[1],
            #                           output_shape=(self.num_actions,))
            value_hidden = Dense(512, activation = 'relu', name = 'value_fc')(hidden_feature)
            value = Dense(1, name = "value")(value_hidden)
            action_hidden = Dense(512, activation = 'relu', name = 'action_fc')(hidden_feature)
            action = Dense(self.num_actions, name = "action")(action_hidden)
            action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keepdims = True), name = 'action_mean')(action) 
            q_value_prediction = Lambda(lambda x: x[0] + x[1] - x[2], name = 'duel_output')([action, value, action_mean])
        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        #select_q_value_of_action = merge([q_value_prediction, action_one_hot], mode='mul',
        #                                 output_shape=(self.num_actions,))

        target_q_value = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=lambda_out_shape)(
            select_q_value_of_action)

        model = Model(inputs=[input_voxel_frame, action_one_hot], outputs=[q_value_prediction, target_q_value])

        # MSE loss on target_q_value only
        model.compile(loss=['mse', 'mse'], loss_weights=[0.0, 1.0], optimizer=Adam(lr=0.00001))  # self.opt)
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def run(self, state, action, reward, terminal, observation):
        next_state = observation

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > self.num_replay_memory:
            self.replay_memory.popleft()

        if self.t >= self.initial_replay_size:
            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.target_network.set_weights(self.q_network.get_weights())

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.save_network_path + '/' + self.exp_name + '_' + str(self.t) + '.h5'
                self.q_network.save(save_path)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_network.predict([np.expand_dims(state[0], axis=0), np.expand_dims(state[1][:3], axis=0),np.expand_dims(state[1][3:], axis=0) ,self.dummy_input])[0])
        self.duration += 1

        if terminal:
            # Observe the mean of rewards on last 30 episode
            self.last_30_reward.append(self.total_reward)
            if len(self.last_30_reward) > 30:
                self.last_30_reward.popleft()

            # Log message
            if self.t < self.initial_replay_size:
                mode = 'random'
            elif self.initial_replay_size <= self.t < self.initial_replay_size + self.exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / AVG_REWARD: {4:2.3f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                    self.episode + 1, self.t, self.duration, self.epsilon,
                    self.total_reward, self.total_q_max / float(self.duration),
                    self.total_loss / (float(self.duration) / float(self.train_interval)), mode))
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / AVG_REWARD: {4:2.3f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                    self.episode + 1, self.t, self.duration, self.epsilon,
                    self.total_reward, self.total_q_max / float(self.duration),
                    self.total_loss / (float(self.duration) / float(self.train_interval)), mode), file=self.log)

            # Init for new game
            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

    def train_network(self):
        state_batch_frame = []
        state_batch_position = []
        state_batch_pose = []
        action_batch = []
        reward_batch = []
        next_state_batch_frame = []
        next_state_batch_position = []
        next_state_batch_pose = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch_frame.append(data[0][0])
            state_batch_position.append(data[0][1][:3])
            state_batch_pose.append(data[0][1][3:])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch_frame.append(data[3][0])
            next_state_batch_position.append(data[3][1][:3])
            next_state_batch_pose.append(data[3][1][3:])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        # Q value from target network
        target_q_values_batch = self.target_network.predict([list2np(next_state_batch_frame),list2np(next_state_batch_position),list2np(next_state_batch_pose), self.dummy_batch])[0]

        # create Y batch depends on dqn or ddqn
        if True: #self.ddqn:
            next_action_batch = np.argmax(self.q_network.predict([list2np(next_state_batch_frame),list2np(next_state_batch_position),list2np(next_state_batch_pose), self.dummy_batch])[0],
                                          axis=-1)
            for i in range(self.batch_size):
                y_batch.append(reward_batch[i] + (1 - terminal_batch[i]) * self.gamma * target_q_values_batch[i][
                    next_action_batch[i]])
            y_batch = list2np(y_batch)
        else:
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=-1)

        a_one_hot = np.zeros((self.batch_size, self.num_actions))
        for idx, ac in enumerate(action_batch):
            a_one_hot[idx, ac] = 1.0

        loss = self.q_network.train_on_batch([list2np(state_batch_frame),list2np(state_batch_position),list2np(state_batch_pose), a_one_hot], [self.dummy_batch, y_batch])

        self.total_loss += loss[1]


def list2np(in_list):
    return np.float32(np.array(in_list))


def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)
