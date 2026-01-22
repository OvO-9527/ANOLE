import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import numpy as np


from simulator.constants import (HD_REWARD, M_IN_K, VIDEO_BIT_RATE)


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










