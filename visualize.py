from sklearn import tree
import pydotplus

def tree_to_png(decisiontree,out_file_name):
    dot_data = tree.export_graphviz(decisiontree,
                                out_file = None,
                                label = 'root',
                                filled = True,
                                impurity=False)

    graph = pydotplus.graph_from_dot_data(dot_data)

    graph.write_png(f'{out_file_name}.png')