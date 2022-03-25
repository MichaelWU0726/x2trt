import pydot

graphs = pydot.graph_from_dot_file("EngineLayers_0.dot")
graph = graphs[0]
graph.write_png("trt_engine.png")