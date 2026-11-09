import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import numpy as np


from simulator.master_QoE_4G.constants import (HD_REWARD, M_IN_K, VIDEO_BIT_RATE)




def QoE_1(current_bitrate: int, last_bitrate: int, rebuffer: float):
    """ Return linear QoE metric.
    平衡QoE
    """
    reward = VIDEO_BIT_RATE[current_bitrate] / M_IN_K - 30 * rebuffer - \
        1 * np.abs(VIDEO_BIT_RATE[current_bitrate] - VIDEO_BIT_RATE[last_bitrate]) / M_IN_K
    return reward



def QoE_2(current_bitrate: int, last_bitrate: int, rebuffer: float):
    """ Return linear QoE metric.
            侧重卡顿QoE
            """

    reward = VIDEO_BIT_RATE[current_bitrate] / M_IN_K - 90 * rebuffer - \
        1 * np.abs(VIDEO_BIT_RATE[current_bitrate] - VIDEO_BIT_RATE[last_bitrate]) / M_IN_K
    return reward

def QoE_3(current_bitrate: int, last_bitrate: int, rebuffer: float):
    """ Return linear QoE metric.
            侧重于高视频质量QoE
            """

    reward = HD_REWARD[current_bitrate]  - 68 * rebuffer - \
        1 * np.abs(HD_REWARD[current_bitrate] - HD_REWARD[last_bitrate])
    return reward


def QoE_4(current_bitrate: int, last_bitrate: int, rebuffer: float):
    """ Return linear QoE metric.
                log的QoE
                """

    log_bit_rate = np.log(VIDEO_BIT_RATE[current_bitrate] / \
                          float(VIDEO_BIT_RATE[0]))
    log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bitrate] / \
                               float(VIDEO_BIT_RATE[0]))
    reward = log_bit_rate \
             - 5.52 * rebuffer \
             - 1 * np.abs(log_bit_rate - log_last_bit_rate)
    return reward

def QoE_5(current_bitrate: int, last_bitrate: int, rebuffer: float):
    """ Return linear QoE metric.
    平衡QoE
    """
    reward = VIDEO_BIT_RATE[current_bitrate] / M_IN_K - 30 * rebuffer - \
        3 * np.abs(VIDEO_BIT_RATE[current_bitrate] - VIDEO_BIT_RATE[last_bitrate]) / M_IN_K
    return reward





def QoE_6(current_bitrate: int, last_bitrate: int, broad_rebuf: float, delay: float,get_vidro_delay:float,broad_buffer_delay:float):
    """ Return linear QoE metric.
        侧重于延迟QoE
        """


    totao_delay = delay/1000 + get_vidro_delay/1000 + broad_buffer_delay

    if totao_delay < 1.1:
        omiga = 0.25
    else:
        omiga = 0.5


    reward = 2 * VIDEO_BIT_RATE[current_bitrate] / M_IN_K - 50 * broad_rebuf - omiga * totao_delay - \
        0.02 * np.abs(VIDEO_BIT_RATE[current_bitrate] - VIDEO_BIT_RATE[last_bitrate]) / M_IN_K
    return reward


def QoE_7(current_bitrate: int,  delay: float, get_vidro_delay:float):
    """ Return linear QoE metric.
        侧重于识别率的QoE
        """
    total_delay = delay/1000 + get_vidro_delay/1000


    reward = 5 * (0.985- 152.00454605826405 / (VIDEO_BIT_RATE[current_bitrate] + 35.947447102074726 )) - 1 * total_delay

    return reward






def load_trace(cooked_trace_folder):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + '/'+ cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names











# for test
def load_traces(cooked_trace_folder):

    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for subdir ,dirs ,files in os.walk( cooked_trace_folder ):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
            file_path = subdir + os.sep + file
            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            cooked_time = []
            cooked_bw = []
            with open(file_path, 'rb') as phile:
                #print(file_path)
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append( file)

    return all_cooked_time, all_cooked_bw, all_file_names










