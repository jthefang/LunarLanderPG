"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network
"""
import os
import tensorflow as tf
import easy_tf_log
import numpy as np
import math
import time
from tensorflow.python.framework import ops

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoints_dir_path = os.path.join(curr_dir_path, 'checkpoints')
log_dir_path = os.path.join(curr_dir_path, 'logs')
historical_log_dir = os.path.join(curr_dir_path, 'historical_logs')

def remove_files_in_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def move_files_in_dir_to(dir_path, dest_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        dest_file_path = os.path.join(dest_path, filename)
        os.rename(file_path, dest_file_path)

def create_dirs():
    if not os.path.exists(checkpoints_dir_path):
        os.makedirs(checkpoints_dir_path)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    if not os.path.exists(historical_log_dir):
        os.makedirs(historical_log_dir)

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        num_episodes=5001,
        load_file_name=None,
        save_file_name="model.ckpt"
    ):

        self.n_x = n_x #input size (= observation space)
        self.n_y = n_y #output size (= action space)
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.gamma = reward_decay

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        self.cost_history = []

        self.build_network()

        self.sess = tf.Session()
        self.init_logging()
        self.sess.run(tf.global_variables_initializer())
        self.load_and_save(load_file_name, save_file_name)

    def init_logging(self):
        create_dirs()
        # $ tensorboard --logdir=logs --port=6006
        # then go to http://acai.local:6006/#scalars&run=.
        self.rewards_dir_path = os.path.join(curr_dir_path, 'logs', 'rewards')
        self.log_dir_path = os.path.join(curr_dir_path, 'logs', 'train')
        self.save_historical_logs()

        #using easy_tf_log
        self.logger = easy_tf_log.Logger()
        self.logger.set_log_dir(self.rewards_dir_path)
        
        #using std tensorboard method
        self.log_every_n = 10
        self.train_writer = tf.summary.FileWriter("logs/train/", flush_secs=5) #self.sess.graph)

    def save_historical_logs(self):
        prev_logs = [filename for filename in os.listdir(self.log_dir_path)]
        if len(prev_logs) > 0:
            self.historical_logs_path = os.path.join(historical_log_dir, 'log-' + str(time.time()))
            self.historical_log_dir = os.path.join(self.historical_logs_path, 'train')
            self.historical_rewards_dir = os.path.join(self.historical_logs_path, 'rewards')
            os.makedirs(self.historical_logs_path)
            os.makedirs(self.historical_log_dir)
            os.makedirs(self.historical_rewards_dir)
            move_files_in_dir_to(self.log_dir_path, self.historical_log_dir)
            move_files_in_dir_to(self.rewards_dir_path, self.historical_rewards_dir)

    def load_and_save(self, load_file_name, save_file_name):
        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        self.save_path = os.path.join(checkpoints_dir_path, save_file_name)
        self.save_every_n = 250 #number of episodes to save weights

        if load_file_name is None:
            self.load_path = tf.train.latest_checkpoint(checkpoints_dir_path) 
        else: 
            load_path = os.path.join(checkpoints_dir_path, load_file_name)
            if os.path.exists(load_path): #checkpoint exists => can load
                self.load_path = load_path
            else: 
                print("Load path {} does not exist".format(load_path))
                return
        if self.load_path is not None:
            self.saver.restore(self.sess, self.load_path)
        print("Loaded checkpoint from {}".format(self.load_path))

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)

    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        observation = observation[:, np.newaxis]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self, episode):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        episode_rewards_sum = sum(self.episode_rewards)
        self.logger.logkv('rewards/episode_rewards_sum', episode_rewards_sum)

        # Train on episode
        summaries, _ = self.sess.run([self.summaries, self.train_op], feed_dict={
             self.X: np.vstack(self.episode_observations).T,
             self.Y: np.vstack(np.array(self.episode_actions)).T,
             self.step: episode,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # log to tensorboard
        if episode % self.log_every_n == 0:
            self.train_writer.add_summary(summaries, episode)
            print("Logged summaries")

        # Save checkpoint
        if (self.save_path is not None) and (episode % self.save_every_n == 0): #save weights every_n episodes
            save_path = self.saver.save(self.sess, self.save_path, global_step=episode)
            print("Model saved in file: %s" % save_path)

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards, self.stock_episode_rewards  = [], [], [], []

        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.step = tf.placeholder(tf.int32)
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        units_layer_1 = 50
        units_layer_2 = 25
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1,self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.softmax(Z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        #log to tensorboard
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            self.loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss
            #one data point
            name = 'loss'
            tf.summary.scalar(name, self.loss)

        #merge data points to one summary variable
        self.summaries = tf.summary.merge_all()

        with tf.name_scope('train'):
            self.lr = 0.0001 + tf.train.exponential_decay(self.learning_rate, self.step, self.num_episodes, 1/math.e) 
            #decays exponentially in our num_epsiodes iterations from self.learning_rate --> .0001
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
