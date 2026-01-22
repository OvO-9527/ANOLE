import matplotlib.pyplot as plt
import numpy as np

# 定义颜色、花纹等
colors = ['#bbdbb3', '#91CCc0', '#f7ac53', '#9998ff', '#7fabd1', '#db7272']
hatch_patterns = ['/', '\\', '-', '.', 'x', '+']
x_labels = ['0-7', '7-14', '14-21', '21-28', '28-35', '>35']


# schemes_show 和 schemes_label 需要你自己定义好
# 例子：

schemes_label = ['MPC', 'Pensieve', 'Merina', 'Netllm', 'Genet', 'ANOLE']

# 四组数据
persent= [[33, 47, 77, 73, 80, 83], [-33, 67, 77, 78, 80, 70],
		  [-14, 59, 73, 71, 72, 79],[-27, 68, 78, 76, 82, 72],
		  [-5, 45, 75,80,78,81],    [58,76,83,83,84.5,86]]
persent_err = [[7.95, 5.81, 4.72, 4.19, 2.46, 5.6], [7.55, 4.54, 3.68, 4.07, 2.3, 5.42],
				   [7.08, 6.58, 4.55, 3.96, 3.18, 6.38], [6.24, 4.55, 2.57, 4.99, 2.27, 7.9],
				   [7.24, 5.33, 4.13, 3.49, 3.22, 6.26], [7.3, 5.71, 3.03, 2.8, 3.32, 3.02]]

qua = [[5.24,9.51,16.62,21.73,28.28,35.81], [4.95,9.49,16.17,21.54,27.47,32.71],
				  [5.35,10.21,16.62,22.07,28.58,34.92], [5.06,9.53,15.94,21.06,27.15,32.24],
				  [4.56,8.91,15.6,21.05,26.2,34.33], [4.96,9.38,16.21,21.89,28.26,35.01]]
qua_err =  [ [0.85, 0.26, 0.43, 0.6, 0.59, 2.9], [0.84, 0.26, 0.43, 0.62, 0.82, 2.05],
					[0.86, 0.26, 0.42, 0.58, 0.63, 2.48], [0.76, 0.22, 0.47, 0.73, 0.88, 2.57],
					 [0.54, 0.26, 0.45, 0.69, 0.82, 1.88],  [0.59, 0.26, 0.5, 0.62, 0.74, 1.24]]

rebuf= [[4.72,4.0,2.23,2.39,1.79,1.74],[6.26,2.81,1.82,2.1,1.76,0.83],
					[7.53,3.68,3.44,2.67,2.18,1.12], [6.66,2.63,1.4,2.03,1.36,0.66],
					[5.23,2.53,1.4,1.93,1.32,0.41], [2.56,1.46,1.12,1.08,0.84,0.16]]
rebuf_err = [[1.98, 0.8, 0.93, 0.68, 0.48, 0.66], [1.42, 0.64, 0.74, 0.83, 0.44, 0.31],
				   [1.09, 0.83, 0.45, 0.56,0.34, 0.18], [1.35, 0.98, 0.62, 0.9, 0.38, 0.34],
				   [1.56, 0.41, 0.64, 0.65, 0.17, 0.22], [1.11, 0.33, 0.43, 0.33, 0.26, 0.1]]


# 创建1行3列子图
fig, axs = plt.subplots(1, 3, figsize=(29, 7))
width = 0.14
x = np.arange(1, 7)
# 绘制第一个子图：Normalize QoE
for scheme in range(len(schemes_label)):
    axs[0].bar(x + (scheme - len(schemes_label) / 2) * width, persent[scheme], width,
               label=schemes_label[scheme], color=colors[scheme], hatch=hatch_patterns[scheme],
               edgecolor='black', yerr=persent_err[scheme], capsize=10)
axs[0].set_ylim(-45, 100)
axs[0].set_xticks(x- 0.1)
axs[0].set_xticklabels(x_labels, fontsize=22)
axs[0].set_ylabel('Normalize QoE', fontsize=22)
axs[0].set_xlabel('Throughput level(Mbps)', fontsize=22)
axs[0].tick_params(axis='both', labelsize=22)
for i in range(len(x) - 1):
    axs[0].plot([x[i] + 4.2 * 0.1, x[i] + 4.2 * 0.1], [-45, 100], linestyle="--", color='grey', linewidth=2)
axs[0].legend(fontsize=20, loc='lower right',ncol=3, handletextpad=0.3, columnspacing=0.6)

# 绘制第二个子图：Bitrate
for scheme in range(len(schemes_label)):
    axs[1].bar(x + (scheme - len(schemes_label) / 2) * width, qua[scheme], width,
               label=schemes_label[scheme], color=colors[scheme], hatch=hatch_patterns[scheme],
               edgecolor='black', yerr=qua_err[scheme], capsize=10)
axs[1].set_ylim(0, 43)
axs[1].set_xticks(x-0.1)
axs[1].set_xticklabels(x_labels, fontsize=22)
axs[1].set_ylabel('Bitrate(Mbps)', fontsize=22)
axs[1].set_xlabel('Throughput level(Mbps)', fontsize=22)
axs[1].tick_params(axis='both', labelsize=22)
for i in range(len(x) - 1):
    axs[1].plot([x[i] + 4.2 * 0.1, x[i] + 4.2 * 0.1], [0, 43], linestyle="--", color='grey', linewidth=2)
axs[1].legend(fontsize=20, loc='upper left',ncol=2, handletextpad=0.3, columnspacing=0.6,bbox_to_anchor=(0.05, 0.9))

# 绘制第三个子图：Rebuffer
for scheme in range(len(schemes_label)):
    axs[2].bar(x + (scheme - len(schemes_label) / 2) * width, rebuf[scheme], width,
               label=schemes_label[scheme], color=colors[scheme], hatch=hatch_patterns[scheme],
               edgecolor='black', yerr=rebuf_err[scheme], capsize=10)
axs[2].set_ylim(0, 9)
axs[2].set_xticks(x-0.1)
axs[2].set_xticklabels(x_labels, fontsize=22)
axs[2].set_ylabel('Rebuffer(sec)', fontsize=22)
axs[2].set_xlabel('Throughput level(Mbps)', fontsize=22)
axs[2].tick_params(axis='both', labelsize=22)
for i in range(len(x) - 1):
    axs[2].plot([x[i] + 4.2 * 0.1, x[i] + 4.2 * 0.1], [0, 9], linestyle="--", color='grey', linewidth=2)
axs[2].legend(fontsize=20, loc='upper right',ncol=2, handletextpad=0.3, columnspacing=0.6,bbox_to_anchor=(0.95, 0.9))


plt.subplots_adjust(wspace=0.2, hspace=0.25)
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.13)

plt.savefig( "F_1.svg", dpi=600)

