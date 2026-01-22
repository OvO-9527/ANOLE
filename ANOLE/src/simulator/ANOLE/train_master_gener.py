import argparse
import os
import time
import warnings
import shutil
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../simulator')))

from sklearn.cluster import KMeans
from simulator.save_cmd import set_seed, save_args
import subprocess
from simulator.ANOLE.master_gener import Pensieve
from simulator.schedulers import (
    UDRTrainScheduler,
)
from simulator.utils import load_traces
from simulator.abr_trace import AbrTrace
from simulator.constants import (
    A_DIM,
    MASTER_EPOCH,
    GER_EPOCH_2,
)


warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


# Saved directory for throughput clustering
output_base_dir = "/home/ubuntu/Load_trace/Cluster_6/"
# Experience pool
master_save_dir="/home/ubuntu/ANOLE/results/master/"



def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument("--save-dir",type=str,required=True,help="direcotry to save the model.")
    parser.add_argument("--seed", type=int, default=30, help="seed")
    parser.add_argument("--gener-2", type=int, default=0, help="General Training Phase 3")
    parser.add_argument("--entropy-weight", type=float, default=np.log(A_DIM), help="Entropy weight")
    parser.add_argument("--master-train", type=int,default=0,help="Training Phase 2.")
    parser.add_argument("--log", action="store_true", help="Use logarithmic form QoE metric")
    parser.add_argument( "--total-epoch",type=int,default=100,help="Total number of epoch to be trained.")
    parser.add_argument("--model-path",type=str,default="",help="Path to a pretrained Tensorflow checkpoint.")
    parser.add_argument("--master-trace-path", type=str, default="", help="Path to a pretrained Tensorflow checkpoint.")
    parser.add_argument("--video-size-file-dir",type=str,default="",help="Path to video size files.")
    parser.add_argument("--nagent",type=int,default=10,help="Path to a pretrained Tensorflow checkpoint.")
    parser.add_argument("--val-freq",type=int,default=700,help="specify to enable validation.")
    parser.add_argument( "--train-trace-dir",type=str,default="",help="A directory contains the training trace files.")
    parser.add_argument( "--val-trace-dir", type=str, default="", help="A directory contains the validation trace files.")



    return parser.parse_args()


def get_max_model(model_result_dir):


    #获取最优模型的路径
    model_result_file = model_result_dir + '/log_val'

    max_row_first_column = None
    #获取最优模型的路径
    with open(model_result_file, 'r') as file:
        lines = file.readlines()

        max_value = None
        # 去除第一行
        lines = lines[2:]

        for line in lines:
            columns = line.strip().split('\t')

            # 获取第四列的值并转换为浮点数
            fourth_column_value = float(columns[3])

            if max_value is None or fourth_column_value > max_value:
                max_value = fourth_column_value
                max_row_first_column = columns[0]

    if max_row_first_column is None:
        print("获取数据错误")
        return ""
    else:
        model_file_path = model_result_dir + '/model_saved/nn_model_ep_' + str(max_row_first_column) + '.ckpt'
        return model_file_path



def main():
    args = parse_args()
    assert (
        not args.model_path
        or args.model_path.endswith(".ckpt")
    )
    os.makedirs(args.save_dir, exist_ok=True)
    print(args.save_dir)
    save_args(args, args.save_dir)
    set_seed(args.seed)

    # Initialize model and agent policy
    pensieve = Pensieve(args.model_path, train_mode=True)

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
    t_start_1 = time.time()

    # # Stage One
    ger_model,entropy_weight = pensieve.train(
        args,
        train_scheduler,
        val_traces,
        args.save_dir,
        args.nagent,
        args.total_epoch,
        args.video_size_file_dir,
        args.log,
        args.val_freq,
        args.master_train,
        args.entropy_weight,
        args.gener_2,
        )

    # Stage two
    if  not args.master_train:
        data_list = []
        cluster_num_1 = 1
        cluster_num_2 = 2
        # Number of clusters
        zhonglei = 6

        # Clustering the training data set
        for filename in os.listdir(args.train_trace_dir):
            try:
                parts = filename.split("_")
                if len(parts) == 3:
                    avg_value = float(parts[0])  # 平均值
                    cov = float(parts[1]) * 25  # CoV
                    index = int(parts[2])  # 序号

                    # 存入列表
                    data_list.append([filename, avg_value, cov, index])
            except Exception as e:
                print(f"跳过文件 {filename}，解析错误: {e}")

        categorys = np.array([[row[cluster_num_1],row[cluster_num_2]] for row in data_list])

        kmeans = KMeans(n_clusters=zhonglei, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(categorys)


        for i in range(len(data_list)):
            data_list[i].append(clusters[i])

        for row in data_list:
            filename = row[0]
            cluster_id = row[-1]

            cluster_dir = os.path.join(output_base_dir, f"{cluster_id + 1}")
            os.makedirs(cluster_dir, exist_ok=True)  # 确保目录存在
            # 复制文件
            src_path = os.path.join(args.train_trace_dir, filename)
            dst_path = os.path.join(cluster_dir, filename)

            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"复制 {filename} 失败: {e}")

        # Stage Two
        #In order to speed up training, multiple specialist training courses are run in parallel.
        processes = []
        for kk in range(zhonglei):
            ma_save_dir = master_save_dir + str(kk+1)
            master_train_trace_dir = output_base_dir + str(kk+1)
            cmd = "python simulator/ANOLE/train_master_QoE.py " \
                  "--total-epoch={total_epoch} " \
                  "--seed={seed} " \
                  "--save-dir={save_dir} " \
                  "--model-path={model_path} " \
                  "--val-freq={val_freq} " \
                  "--qoe-number={qoe_number} " \
                  "--entropy-weight={entropy_weight} " \
                  "--master-train={master_train} " \
                  "--nagent={nagent} ".format(
                total_epoch=MASTER_EPOCH,
                seed=30,
                save_dir=ma_save_dir,
                model_path=ger_model,
                val_freq=700,
                qoe_number= 1,
                entropy_weight=entropy_weight,
                master_train=1,
                nagent=5)

            if args.log:
                cmd += "--log "

            cmd += "--val-trace-dir={val_dir} " \
                   "--train-trace-dir={train_trace_dir}".format(
                val_dir=master_train_trace_dir,
                train_trace_dir=master_train_trace_dir)
            print(cmd)
            processes.append(subprocess.Popen(cmd.split(' ')))
            #subprocess.run(cmd.split(' '))

        for process in processes:
            process.wait()

       #Access a pool of specialist experience
        for i in range(zhonglei):
            model_result_dir = master_save_dir + str(i+1)
            dest_QoE_dir = args.master_trace_path
            master_train_trace_dir = output_base_dir + str(i + 1)
            model_file_path =  get_max_model(model_result_dir)
            print(model_file_path)
            pensieve.master_trace_test(model_file_path,master_train_trace_dir,dest_QoE_dir,args.video_size_file_dir,1)



    # Stage Three
    if not args.master_train :
        time_start = time.time()
        save_dir_2 = args.save_dir + '_' +str(2)
        cmd = "python simulator/ANOLE/train_master_gener.py " \
              "--total-epoch={total_epoch} " \
              "--seed={seed} " \
              "--gener-2={gener_2} " \
              "--save-dir={save_dir} " \
              "--model-path={model_path} " \
              "--val-freq={val_freq} " \
              "--entropy-weight={entropy_weight} " \
              "--master-train={master_train} " \
              "--nagent={nagent} ".format(
            total_epoch=GER_EPOCH_2,
            seed=30,
            gener_2=1,
            save_dir=save_dir_2,
            model_path=ger_model,
            val_freq=700,
            entropy_weight=entropy_weight,
            master_train=1,
            nagent=20)
        if args.log:
            cmd += "--log "
        cmd += "--val-trace-dir={val_dir} " \
               "--train-trace-dir={train_trace_dir} " \
               "--master-trace-path={master_trace_path}".format(
            val_dir=args.val_trace_dir,
            train_trace_dir=args.train_trace_dir,
            master_trace_path = args.master_trace_path)
        print(cmd)
        subprocess.run(cmd.split(' '))



if __name__ == "__main__":
    main()

