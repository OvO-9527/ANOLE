import os

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn



class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, bitrate_dim, entropy_weight,name = 'actor'):

    def create_actor_network(self):
        with tf.compat.v1.variable_scope(self.scope_name,reuse=tf.compat.v1.AUTO_REUSE):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            return inputs, out

    def train_1(self, inputs,pro_old, acts, Adv1):

        self.sess.run(self.actor_optimize_1, feed_dict={
            self.inputs: inputs,
            self.pro_old_tensor: pro_old,
            self.acts: acts,
            self.Adv1: Adv1,
            self.entropy_weight: 0.1,
        })

        return self._entropy_weight



    def train_2(self, inputs,pro_old, acts, Adv1, master_s_1, master_a_1):

        self.sess.run(self.actor_optimize_2, feed_dict={
            self.inputs: inputs,
            self.pro_old_tensor: pro_old,
            self.master_inputs1: master_s_1,
            self.master_acts1: master_a_1,
            self.acts: acts,
            self.Adv1: Adv1,
            self.entropy_weight: 0.1,
        })

        self._entropy_weight


    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })


    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate, bitrate_dim, name = 'critic'):

    def create_critic_network(self):
        with tf.compat.v1.variable_scope(self.scope_name):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            return inputs, out


    def train(self, inputs, td_target1):
        self.sess.run(self.critic_optimize, feed_dict={
            self.inputs: inputs,
            self.td_target1: td_target1,

        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })


    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients_actor(s_batch, pro_old,  acts, Gae, Adv, actor, entropy_weight):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """


    actor.train(s_batch, pro_old, acts,Gae, Adv, entropy_weight)



def compute_gradients_critic(s_batch, R_batch, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """

    critic.train(s_batch, R_batch)





def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


