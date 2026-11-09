MODEL_SAVE_INTERVAL = 700

VIDEO_BIT_RATE = [200., 800., 2200., 5000., 10000.,18000.,32000.,50000.]  # Kbps

HD_REWARD = [0.77,2.1,13.1,22.3,38.8, 65.2, 111.4, 170.8]
M_IN_K = 1000.0

REBUF_PENALTY = 30  # 1 sec rebuffering -> 3 Mbps
REBUF_penalty = 5.52

SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent


# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and
# time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 8
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
BUFFER_NORM_FACTOR = 10.0
RAND_RANGE = 1000
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size
# download_time reward
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNK_LEN = 4000.0  # millisec, every time add this amount to buffer
TOTAL_VIDEO_CHUNK = 48
# TOTAL_VIDEO_CHUNK = 48
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
TRAIN_SEQ_LEN = 100  # batchsize of pensieve training
SAMPLE_LEN = 50  # batchsize of sampoling
VIDEO_SIZE_FILE = '../data/video_sizes_sync/video_size_'
BITRATE_DIM = 8

NUM_VARIANTS = 30
GAE_gamma = 0.95


MASTER_EPOCH = 10000
GER_EPOCH_2 = 15000