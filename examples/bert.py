import taso as ts
import onnx
import numpy as np

seq_length = 64
hidden_dims = 1024
batch_size = 16

def attention(graph, input, heads):
    d_model = input.dim(1)
    d_k = d_model // heads
    assert input.dim(1) % heads == 0
    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(d_model, d_model)))
    # compute query, key, value tensors
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(batch_size, 64,16,64))
    k = graph.reshape(k, shape=(batch_size, 64,16,64))
    v = graph.reshape(v, shape=(batch_size, 64,16,64))
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(0,2,1,3), shuffle=True)
    k = graph.transpose(k, perm=(0,2,1,3), shuffle=True)
    v = graph.transpose(v, perm=(0,2,1,3), shuffle=True)
    # perform matrix multiplications
    logits = graph.matmul(q, k)
    output = graph.matmul(logits, v)
    # transpose the output back
    output = graph.transpose(output,perm=(0,2,1,3), shuffle=True)
    output = graph.reshape(output, shape=(batch_size * 64, 1024))

    # a final linear layer
    linear = graph.new_weight(dims=(d_model, d_model))
    output = graph.matmul(output, linear)
    return output

avg_runtime_ts = np.zeros(10)
avg_cost_ts = np.zeros(10)
avg_runtime_baseline = np.zeros(10)
avg_cost_baseline = np.zeros(10)

for i in range(10):
    graph = ts.new_graph()
    input = graph.new_input(dims=(batch_size * seq_length, hidden_dims))
    input = graph.relu(input)
    t = input
    for i in range(8):
        t = attention(graph, t, 16)

    avg_runtime_baseline[i] = graph.run_time()
    avg_cost_baseline[i] = graph.cost()
    
    new_graph = ts.optimize(graph, alpha=1.05, budget=1000)
    avg_runtime_ts[i] = new_graph.run_time()
    avg_cost_ts[i] = new_graph.cost()

    onnx_model = ts.export_onnx(new_graph)
    onnx.checker.check_model(onnx_model)
    #onnx.save(onnx_model, "bert_taso.onnx")

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
