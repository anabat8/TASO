import taso
import onnx
import numpy as np

hidden_size = 512
length = 5

def combine(graph, x, h):
    w1 = graph.new_weight(dims=(hidden_size, x.dim(1)))
    w2 = graph.new_weight(dims=(hidden_size, h.dim(1)))
    return graph.add(graph.matmul(x, w1), graph.matmul(h, w2))

def nas_node(graph, input, x):
    t = list()
    for i in range(8):
        t.append(combine(graph, x, input))
    midt = list()
    midt.append(graph.add(graph.relu(t[0]), graph.sigmoid(t[3])))
    midt.append(graph.add(graph.sigmoid(t[1]), graph.tanh(t[2])))
    midt.append(graph.mul(graph.sigmoid(t[4]), graph.tanh(t[5])))
    midt.append(graph.mul(graph.sigmoid(t[6]), graph.relu(t[7])))
    midt.append(graph.add(graph.sigmoid(midt[1]), graph.tanh(midt[2])))
    midt.append(graph.mul(graph.tanh(midt[0]), graph.tanh(midt[3])))
    midt.append(graph.mul(graph.tanh(midt[4]), graph.tanh(midt[5])))
    return graph.tanh(midt[6])

avg_runtime_ts = np.zeros(10)
avg_cost_ts = np.zeros(10)
avg_runtime_baseline = np.zeros(10)
avg_cost_baseline = np.zeros(10)

for run in range(10):
    graph = taso.new_graph()
    xs = list()
    for i in range(length):
        xs.append(graph.new_input(dims=(1, hidden_size)))
    state = graph.new_weight(dims=(1, hidden_size))
    for i in range(length):
        state = nas_node(graph, state, xs[i])

    avg_runtime_baseline[run] = graph.run_time()
    avg_cost_baseline[run] = graph.cost()
    
    new_graph = taso.optimize(graph, alpha=1.05, budget=1000)
    avg_runtime_ts[run] = new_graph.run_time()
    avg_cost_ts[run] = new_graph.cost()

    #onnx_model = taso.export_onnx(new_graph)
    #onnx.checker.check_model(onnx_model)
    #onnx.save(onnx_model, "nasrnn_taso.onnx")

graph_runtime = avg_runtime_baseline.mean()
graph_cost = avg_cost_baseline.mean()

print("Measuring the performance of computation graph before optimization")
print("End-to-end avg inference time = {}ms".format(graph_runtime))
print("Unoptimized graph avg cost", " = {}".format(graph_cost))

new_graph_runtime = avg_runtime_ts.mean()
new_graph_cost = avg_cost_ts.mean()

print("Measuring the performance of computation graph after optimization")
print("End-to-end avg inference time = {}ms".format(new_graph_runtime))
print("Optimized graph avg cost", " = {}".format(new_graph_cost))
