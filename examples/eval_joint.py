import taso as ts
import onnx

# 1. evaluate the performance by just considering substitution optimizations
print("Measuring the performance of graph substitution optimizations (average of 1000 runs)")
graph = ts.load_onnx('examples/bert_graphs/bert_subst_nw.onnx')
print("Taso: end-to-end inference time = {}ms".format(graph.run_time()))
print()

#2. evaluate the performance by just performing data layout optimizations
print("Measuring the performance of data layout optimizations")
graph = ts.load_onnx('examples/bert_graphs/bert_layout_nw.onnx')
print("Taso: end-to-end inference time = {}ms".format(graph.run_time()))
print()

#3. evaluate the performance by sequential optimizations
print("Measuring the performance of sequential optimizations")
graph = ts.load_onnx('examples/bert_graphs/bert_sequential_nw.onnx')
print("Taso: end-to-end inference time = {}ms".format(graph.run_time()))
print()

#4. evaluate the performance by joint optimizations
print("Measuring the performance of joint optimizations")
graph = ts.load_onnx('examples/bert_graphs/bert_xflow_nw.onnx')
print("Taso: end-to-end inference time = {}ms".format(graph.run_time()))
print()

