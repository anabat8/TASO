import matplotlib.pyplot as plt
import numpy as np

# Inference times in milliseconds
# Format: [TensorFlow, TensorFlow XLA, TASO w/ cuDNN]
inference_times = {
    "ResNet-50":     [2.5, 2.3, 1.8],
    "NasNet-A":      [11.5, 11.7, 8.9],
    "ResNeXt-50":    [19.8, 30.0, 7.0],
    "NasRNN":        [12.8, 10.0, 4.5],
    "BERT":          [2.0, 2.2, 1.4],
}

labels = list(inference_times.keys())
tf_times = [inference_times[m][0] for m in labels]
xla_times = [inference_times[m][1] for m in labels]
taso_times = [inference_times[m][2] for m in labels]

x = np.arange(len(labels))  # Label locations
width = 0.25  # Width of bars

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width, tf_times, width, label='TensorFlow', color='#FF7F0E')
rects2 = ax.bar(x, xla_times, width, label='TensorFlow XLA', color='#FFBB78')
rects3 = ax.bar(x + width, taso_times, width, label='TASO (cuDNN)', color='#1F77B4')

ax.set_ylabel('Inference Time (ms)')
ax.set_title('Inference Time by DNN and Backend')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim([0, max(tf_times + xla_times + taso_times) + 5])

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

fig.tight_layout()
plt.savefig("inference_performance_experiment.png", dpi=300)
plt.show()
