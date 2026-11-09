import csv
import argparse
import logging
import os
import sys
import time
import warnings
import numpy as np
from typing import List
import multiprocessing as mp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

from replay_memory import ReplayMemory
import tensorflow as tf
import shutil
from common.utils import set_seed, save_args
import subprocess
from simulator.master_QoE_4G.schedulers import (
    UDRTrainScheduler,
)
from simulator.master_QoE_4G.abr_trace import AbrTrace
from simulator.master_QoE_4G.base_abr import BaseAbr
from simulator.master_QoE_4G.schedulers import TestScheduler
from simulator.master_QoE_4G.utils import load_traces,load_trace, QoE_1,QoE_2,QoE_3,QoE_4,QoE_5
from simulator.master_QoE_4G.env import Environment
from simulator.master_QoE_4G.env_ZB import Environment as Env_zb
from simulator.master_QoE_4G.ALONE_git2 import A_NN
from simulator.master_QoE_4G.constants import (
    A_DIM,
    BUFFER_NORM_FACTOR,
    CRITIC_LR_RATE,
    ACTOR_LR_RATE,
    DEFAULT_QUALITY,
    S_INFO,
    S_LEN,
    VIDEO_BIT_RATE,
    M_IN_K,
    VIDEO_CHUNK_LEN,
    MILLISECONDS_IN_SECOND,
    TOTAL_VIDEO_CHUNK,
    TRAIN_SEQ_LEN,
    SAMPLE_LEN,
    GAE_gamma,
    NUM_VARIANTS,
    GER_EPOCH_2,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


GAMMA = 0.99
#BITRATE_DIM = 10
RAND_RANGE = 1000
warnings.filterwarnings("ignore")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU:",physical_devices)
for device in physical_devices:

    tf.config.experimental.set_memory_growth(device, True)




def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument("--save-dir",type=str,required=True,help="direcotry to save the model.")
    parser.add_argument("--seed", type=int, default=30, help="seed")
    parser.add_argument("--entropy-weight", type=float, default=np.log(A_DIM), help="Entropy weight")
    parser.add_argument("--master-train", type=int,default=0,help="Total number of epoch to be trained.")
    parser.add_argument("--log", action="store_true", help="Use logarithmic form QoE metric")
    parser.add_argument( "--total-epoch",type=int,default=100,help="Total number of epoch to be trained.")
    parser.add_argument("--model-path",type=str,default="",help="Path to a pretrained Tensorflow checkpoint.")
    parser.add_argument("--video-size-file-dir",type=str,default="",help="Path to video size files.")
    parser.add_argument("--nagent",type=int,default=10,help="Path to a pretrained Tensorflow checkpoint.")
    parser.add_argument("--val-freq",type=int,default=700,help="specify to enable validation.")
    parser.add_argument("--qoe-number",type=int,default=0,help="Serial number of the selected QoE function")
    parser.add_argument( "--train-trace-dir",type=str,default="",help="A directory contains the training trace files.")
    parser.add_argument( "--val-trace-dir", type=str, default="", help="A directory contains the validation trace files.")



    return parser.parse_args()



def calculate_from_selection(selected, last_bit_rate):
    # naive step implementation
    # action=0, bitrate-1; action=1, bitrate stay; action=2, bitrate+1
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    bit_rate = max(0, bit_rate)
    bit_rate = min(9, bit_rate)

    return bit_rate


def test( actor: A_NN.ActorNetwork, trace: AbrTrace,
          video_size_file_dir: str, save_dir: str, is_log: bool,qoe_number:int ):
    os.makedirs(save_dir, exist_ok=True)
    if trace.name:
        log_name = os.path.join(save_dir, "{}.csv".format( trace.name))
    else:
        log_name = os.path.join(save_dir, "log.csv")
    abr_log = open(log_name, 'w')
    log_writer = csv.writer(abr_log, lineterminator='\n')
    log_writer.writerow(["timestamp", "bitrate", "buffer_size",
                         "rebuffering", "video_chunk_size", "delay",
                         "reward"])
    test_scheduler = TestScheduler(trace)


    net_env = Environment(test_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                          video_size_file_dir=video_size_file_dir)
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)

    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    final_reward = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real

        (
            delay,
            sleep_time,
            buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain

        ) = net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms


        # reward is video quality - rebuffer penalty - smoothness


        reward = QoE_1(bit_rate, last_bit_rate, rebuf)



        r_batch.append(reward)

        last_bit_rate = bit_rate

        log_writer.writerow([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                             buffer_size, rebuf, video_chunk_size, delay,
                             reward])

        # log time_stamp, bit_rate, buffer_size, reward
        state = np.array(s_batch[-1], copy=True)
        # dequeue history record
        state = np.roll(state, -1, axis=1)
        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = (
                float(video_chunk_size) / float(delay) / M_IN_K
        )  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, : A_DIM] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
                                   )  # mega byte
        state[5, -1] = np.minimum(
            video_chunk_remain, TOTAL_VIDEO_CHUNK,
        ) / float(TOTAL_VIDEO_CHUNK)

        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        action_cumsum = np.cumsum(action_prob)

        bit_rate = (
                action_cumsum
                > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
        ).argmax()


        s_batch.append(state)

        entropy_record.append(A_NN.compute_entropy(action_prob[0]))

        if end_of_video:

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            final_reward = sum(r_batch)
            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)

            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            break
    abr_log.close()

    return final_reward






def train( trace_scheduler, val_traces: List[AbrTrace],
          save_dir: str, num_agents: int, total_epoch: int,
          video_size_file_dir: str, is_log: bool, model_save_interval: int ,model_path, entropy_weight,qoe_number:int):


    # Visdom Settings
    # Visdom Logs
    print('master train start')
    val_epochs = []
    val_mean_rewards = []
    average_rewards = []
    average_entropies = []
    entropy_weight = entropy_weight

    logging.basicConfig(filename=os.path.join(save_dir, 'log_central'),
                        filemode='w', level=logging.INFO)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(num_agents):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    agents = []


    for i in range(num_agents):
        agents.append(mp.Process(
            target=agent,
            args=(TRAIN_SEQ_LEN, S_INFO, S_LEN, A_DIM,
                  i, net_params_queues[i], exp_queues[i], trace_scheduler,
                  video_size_file_dir, qoe_number ,is_log)))
    for i in range(num_agents):
        agents[i].start()


    with tf.compat.v1.Session() as sess, \
            open(os.path.join(save_dir, 'log_train'), 'w', 1) as log_central_file, \
            open(os.path.join(save_dir, 'log_val'), 'w', 1) as val_log_file:
        log_writer = csv.writer(log_central_file, delimiter='\t', lineterminator='\n')
        log_writer.writerow(['epoch', 'loss', 'avg_reward', 'avg_entropy'])
        val_log_writer = csv.writer(val_log_file, delimiter='\t', lineterminator='\n')
        val_log_writer.writerow(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max'])

        epoch = 0

        actor = A_NN.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN],
                                 action_dim=A_DIM,
                                 learning_rate = ACTOR_LR_RATE,
                                 bitrate_dim=A_DIM,
                                 entropy_weight=entropy_weight,
                                 name='actor'
                                         )

        critic = A_NN.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE,
                                   bitrate_dim=A_DIM,
                                   name='QoE1')


        # 创建一个用于更新的旧actor网络
        model_actor_old = A_NN.ActorNetwork(sess,
                                                   state_dim=[S_INFO, S_LEN],
                                                   action_dim=A_DIM,
                                                   learning_rate=ACTOR_LR_RATE,
                                                   bitrate_dim=A_DIM,
                                                   entropy_weight=entropy_weight,
                                                   name='old_actor')

        logging.info('actor and critic initialized')

        sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver(max_to_keep=None)  # save neural net parameters
        # restore neural net parameters
        if model_path:  # nn_model is the path to file
            saver.restore(sess, model_path)
            print("Model restored.")


        os.makedirs(os.path.join(save_dir, "model_saved"), exist_ok=True)


        val_rewards = [test(
            actor, trace, video_size_file_dir=video_size_file_dir,
            save_dir=os.path.join(save_dir, "val_logs"), is_log=is_log,qoe_number=qoe_number) for trace in val_traces]
        val_mean_reward = np.mean(val_rewards)
        max_avg_reward = val_mean_reward

        val_log_writer.writerow(
            [epoch, np.min(val_rewards),
             np.percentile(val_rewards, 5), np.mean(val_rewards),
             np.median(val_rewards), np.percentile(val_rewards, 95),
             np.max(val_rewards)])
        val_epochs.append(epoch)
        val_mean_rewards.append(val_mean_reward)


        # 创建策略缓冲区和价值缓冲区
        po_buff = ReplayMemory(15 * TRAIN_SEQ_LEN)


        #-----------------------此为正常训练步骤----------------------------------------------------
        while epoch < total_epoch:

            start_t = time.time()
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()

            for i in range(num_agents):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            # in the experiences from the agents


            model_actor_old.set_network_params(actor_net_params)


            # 获取每个进程的轨迹s，a，r
            for i in range(num_agents):
                s_batch, a_batch, Adv_batch_1,R_batch_1 = exp_queues[i].get()
                po_buff.push([s_batch, a_batch,  Adv_batch_1,R_batch_1])



            for i in range(num_agents):
                s_batch, a_batch, Adv_batch_1,R_batch_1 = exp_queues[i].get()
                po_buff.push([s_batch, a_batch,  Adv_batch_1,R_batch_1])




            for _ in range(2):
                s_batch, a_batch,  Adv_batch_1,  R_batch_1 = po_buff.sample(SAMPLE_LEN * 9)
                pro_old = model_actor_old.predict(s_batch)
                entropy_weight = actor.train_1(s_batch, pro_old, a_batch, Adv_batch_1)

                critic.train(s_batch, R_batch_1)




            # # 策略缓存区清零
            po_buff.clear()
            epoch += 1
            print(epoch)


            if epoch % model_save_interval == 0 and epoch!=0:
                # # Visdom log and plot
                val_rewards = [test(
                    actor, trace, video_size_file_dir=video_size_file_dir,
                    save_dir=os.path.join(save_dir, "val_logs"), is_log=is_log,qoe_number=qoe_number) for trace in val_traces]
                val_mean_reward = np.mean(val_rewards)

                val_log_writer.writerow(
                    [epoch, np.min(val_rewards),
                     np.percentile(val_rewards, 5), np.mean(val_rewards),
                     np.median(val_rewards), np.percentile(val_rewards, 95),
                     np.max(val_rewards)])

                save_path = saver.save(
                    sess,
                    os.path.join(save_dir, "model_saved", f"nn_model_ep_{epoch}.ckpt"))
                logging.info("Model saved in file: " + save_path)




    for tmp_agent in agents:
        tmp_agent.terminate()
        tmp_agent.join()





def agent(train_seq_len: int, s_info: int, s_len: int, a_dim: int,
           agent_id: int, net_params_queue: mp.Queue,
          exp_queue: mp.Queue, trace_scheduler, video_size_file_dir: str,
          qoe_number: int, is_log: bool):
    """Agent method for A2C/A3C training framework.

    Args
        train_seq_len: train batch size
        net_params_queues: a queue for the transferring neural network
                           parameters between central agent and agent.
        exp_queues: a queue for the transferring experience/rollouts
                    between central agent and agent.
    """

    if agent_id < 5:
        gpu_id = 0
    else:
        gpu_id = 1
    gpu_id = 0
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
        print(f"Agent {agent_id}: Using GPU {gpu_id}")
    else:
        print(f"Agent {agent_id}: No GPU devices found.")




    net_env = Environment(trace_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                          video_size_file_dir=video_size_file_dir,random_seed=agent_id)



    with tf.compat.v1.Session() as sess:

        actor = A_NN.ActorNetwork(sess, state_dim=[s_info, s_len],
                                 action_dim=a_dim, learning_rate = ACTOR_LR_RATE,bitrate_dim=A_DIM,entropy_weight=np.log(A_DIM),
                                         name = 'actor')
        # learning_rate=args.ACTOR_LR_RATE)
        critic = A_NN.CriticNetwork(sess,
                                                 state_dim=[s_info, s_len],
                                                 learning_rate=CRITIC_LR_RATE,
                                                 bitrate_dim=A_DIM,
                                                 name='QoE')

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params= net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)


        last_bit_rate = DEFAULT_QUALITY
        selection = 0
        bit_rate = DEFAULT_QUALITY

        # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
        action_vec = np.zeros(a_dim)

        action_vec[bit_rate] = 1

        s_batch = [np.zeros((s_info, s_len))]
        a_batch = [action_vec]
        r_batch = []


        entropy_record = []


        time_stamp = 0
        epoch = 0
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real


            (
                delay,
                sleep_time,
                buffer_size,
                rebuf,
                video_chunk_size,
                next_video_chunk_sizes,
                end_of_video,
                video_chunk_remain
            ) = net_env.get_video_chunk(bit_rate)




            reward = QoE_1(bit_rate, last_bit_rate, rebuf)




            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be args.S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                           float(np.max(VIDEO_BIT_RATE))  # last quality

            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                           float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, TOTAL_VIDEO_CHUNK) / float(TOTAL_VIDEO_CHUNK)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(
                state, (1, s_info, s_len)))

            action_cumsum = np.cumsum(action_prob)


            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

            # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(A_NN.compute_entropy(action_prob[0]))

            # report experience to the coordinator
            if len(r_batch) >= train_seq_len or end_of_video:
                epoch += 1
                # ------------------------------回报R和优势td_batch以及广义优势估计GAE----------------------------------
                # 更改格式，便于下面计算
                s_batch_np = np.stack(s_batch, axis=0)
                a_batch_np = np.vstack(a_batch)
                r_batch_np = np.vstack(r_batch)


                assert s_batch_np.shape[0] == a_batch_np.shape[0]
                assert s_batch_np.shape[0] == r_batch_np.shape[0]
                ba_size = s_batch_np.shape[0]
                v_batch = critic.predict(s_batch_np)
                R_batch = np.zeros(r_batch_np.shape)


                if end_of_video:
                    R_batch[-1, 0] = 0  # terminal state

                else:
                    R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
                for t in reversed(range(ba_size - 1)):
                    R_batch[t, 0] = r_batch_np[t] + GAMMA * R_batch[t + 1, 0]
                Adv_batch = R_batch - v_batch
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               Adv_batch[1:],
                               R_batch[1:],
                               ])

                if epoch % 2 == 0 and epoch != 0:
                    # synchronize the network parameters from the coordinator
                    actor_net_params, critic_net_params = net_params_queue.get()
                    actor.set_network_params(actor_net_params)
                    critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros(a_dim)

                action_vec[bit_rate] = 1
                s_batch.append(np.zeros((s_info, s_len)))
                a_batch.append(action_vec)


                net_env.trace_scheduler.set_epoch(epoch)

            else:
                s_batch.append(state)
                # print(bit_rate)
                # action_vec = np.zeros(args.A_DIM)
                # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros(a_dim)

                action_vec[bit_rate] = 1
                # print(action_vec)
                a_batch.append(action_vec)

def main():
    args = parse_args()
    assert (
        not args.model_path
        or args.model_path.endswith(".ckpt")
    )  # 断言模型路径为空或以 .ckpt 结尾
    os.makedirs(args.save_dir, exist_ok=True)
    print(args.save_dir)
    save_args(args, args.save_dir)
    set_seed(args.seed)

    # training_traces, validation_traces,


    training_traces = []
    val_traces = []

    if args.train_trace_dir:
        all_time, all_bw, all_file_names = load_traces(args.train_trace_dir)
        training_traces = [AbrTrace(t, bw, link_rtt=80, buffer_thresh=60, name=name)
                           for t, bw, name in zip(all_time, all_bw, all_file_names)]

    if args.val_trace_dir:
        all_time, all_bw, all_file_names = load_traces(args.val_trace_dir)
        val_traces = [AbrTrace(t, bw, link_rtt=80, buffer_thresh=60, name=name)
                           for t, bw, name in zip(all_time, all_bw, all_file_names)]
    train_scheduler = UDRTrainScheduler(
        training_traces,
        percent=1,
    )

    train(
        train_scheduler,
        val_traces,
        args.save_dir,
        args.nagent,
        args.total_epoch,
        args.video_size_file_dir,
        args.log,
        args.val_freq,
        args.model_path,
        args.entropy_weight,
        args.qoe_number
        )




if __name__ == "__main__":
    t_start = time.time()
    main()
    print("time used: {:.2f}s".format(time.time() - t_start))
