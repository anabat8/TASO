import taso as ts
import onnx
import numpy as np

def resnet_block(graph, input, strides, out_channels):
    w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    w2 = graph.new_weight(dims=(out_channels,t.dim(1),3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    w3 = graph.new_weight(dims=(4*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*4):
        w4 = graph.new_weight(dims=(out_channels*4,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return graph.relu(graph.add(input, t))

avg_runtime_ts = np.zeros(10)
avg_cost_ts = np.zeros(10)
avg_runtime_baseline = np.zeros(10)
avg_cost_baseline = np.zeros(10)

for run in range(10):
    graph = ts.new_graph()
    input = graph.new_input(dims=(1,64,56,56))
    t = input
    for i in range(3):
        t = resnet_block(graph, t, (1,1), 64)
    strides = (2,2)
    for i in range(4):
        t = resnet_block(graph, t, strides, 128)
        strides = (1,1)
    strides = (2,2)
    for i in range(6):
        t = resnet_block(graph, t, strides, 256)
        strides = (1,1)
    strides = (2,2)
    for i in range(3):
        t = resnet_block(graph, t, strides, 512)
        strides = (1,1)

    avg_runtime_baseline[run] = graph.run_time()
    avg_cost_baseline[run] = graph.cost()
    
    new_graph = ts.optimize(graph, alpha=1.05, budget=1000)
    avg_runtime_ts[run] = new_graph.run_time()
    avg_cost_ts[run] = new_graph.cost()

    #onnx_model = ts.export_onnx(new_graph)
    #onnx.checker.check_model(onnx_model)
    #onnx.save(onnx_model, "resnet50_taso.onnx")

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