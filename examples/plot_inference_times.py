import matplotlib.pyplot as plt
import numpy as np

# Inference times in milliseconds
# Format: [TensorFlow, TensorFlow XLA, TASO w/ cuDNN, MetaFlow]
inference_times = {
    "ResNet-50":     [6.663, 5.156, 5.594, 6.533],
    "NasNet-A":      [29.840, 24.386, 12.466, 14.671],
    "ResNeXt-50":    [20.777, 10.402, 4.940, 5.696],
    "NasRNN":        [3.644, 1.299, 0.0, 1.038],
    "BERT":          [4.054, 4.055, 13.819, 15.413],
}

# Graph costs for TASO w/ cuDNN and MetaFlow
# Format: [TASO w/ cuDNN, MetaFlow]
graph_costs = {
    "ResNet-50":     [9.460, 9.554],
    "NasNet-A":      [18.469, 30.017],
    "ResNeXt-50":    [7.803, 9.109],
    "NasRNN":        [0.914, 25.674],
    "BERT":          [23.787, 35.285],
}

labels = list(inference_times.keys())
x = np.arange(len(labels))  # Bar locations
width = 0.2  # Width of each bar

# Prepare bar heights
tf_times = [inference_times[m][0] for m in labels]
xla_times = [inference_times[m][1] for m in labels]
taso_times = [inference_times[m][2] for m in labels]
meta_times = [inference_times[m][3] for m in labels]

# Plot 1: Inference Time
fig1, ax1 = plt.subplots(figsize=(10, 5))
rects1 = ax1.bar(x - 1.5*width, tf_times, width, label='TensorFlow', color='#FF7F0E')
rects2 = ax1.bar(x - 0.5*width, xla_times, width, label='TensorFlow XLA', color='#FFBB78')
rects3 = ax1.bar(x + 0.5*width, taso_times, width, label='TASO (cuDNN)', color='#1F77B4')
rects4 = ax1.bar(x + 1.5*width, meta_times, width, label='MetaFlow', color='#98DF8A')

ax1.set_ylabel('Inference Time (ms)')
ax1.set_title('End-to-end Inference Performance Comparison Among DNN Frameworks')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.set_ylim([0, max(tf_times + xla_times + taso_times + meta_times) + 5])

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)

fig1.tight_layout()
plt.savefig("inference_performance_experiment.png", dpi=300)
plt.show()

# -------------------------
# Plot 2: Graph Costs
# -------------------------

taso_costs = [graph_costs[m][0] for m in labels]
meta_costs = [graph_costs[m][1] for m in labels]

fig2, ax2 = plt.subplots(figsize=(10, 5))
rects5 = ax2.bar(x - 0.2, taso_costs, width, label='TASO (cuDNN)', color='#1F77B4')
rects6 = ax2.bar(x + 0.2, meta_costs, width, label='MetaFlow', color='#98DF8A')

ax2.set_ylabel('Graph Cost')
ax2.set_title('Computation Graph Cost Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.set_ylim([0, max(taso_costs + meta_costs) + 5])

def add_labels_cost(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

add_labels_cost(rects5)
add_labels_cost(rects6)

fig2.tight_layout()
plt.savefig("graph_cost_comparison.png", dpi=300)
plt.show()
