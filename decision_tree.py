import pandas as pd
import numpy as np
import operator
from collections import Counter, defaultdict


class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.condition = None
        self.name = None
        self.label = None
        self.depth = 0
        self.samples = 0
        self.healthy = 0
        self.trisomic = 0


class Tree:
    def __init__(self):
        self.root = Node()

    def find_node(self, node, name):
        if node.name == name:
            return node
        left = self.find_node(node.left, name)
        if left.name == name:
            return left
        right = self.find_node(node.right, name)
        if right.name == name:
            return right
        return None

    def create_node(self, node, side):
        if side == 'right':
            node.right = Node()
            node.right.depth = node.depth + 1
            return node.right
        if side == 'left':
            node.left = Node()
            node.left.depth = node.depth + 1
            return node.left

    def output(self, node):
        if node != None:
            return zip(node.depth, node.name, node.condition)
            # self.output(node.left)
            # self.output(node.right)




def entropy(*probs):
    if not isinstance(probs, tuple):
        probs = [probs]
    probs = np.array(probs)
    logs = np.log2(1. / probs).T
    logs[probs == 0] = 0
    return probs.dot(logs)


def conditional_entropy(attr, attr_name, threshold):
    prob_factor = np.sum([attr[attr_name] < threshold], dtype=np.float) / len(attr.index)
    sub_rows = attr[attr[attr_name] < threshold]
    # sub_prob = np.sum([sub_rows['class_label'] == 0.0], dtype=np.float) / len(sub_rows.index)

    # if len(sub_rows.index) == 0:
    # sub_prob = 0
    if prob_factor == 0:
        intermidiate_res = 0
    else:
        label = set(sub_rows['class_label'].values).pop()
        sub_prob = np.sum((sub_rows['class_label'] == label).values, dtype=np.float) / len(sub_rows.index)
        intermidiate_res = prob_factor * entropy(sub_prob, (1 - sub_prob))

    sub_rows = attr[attr[attr_name] >= threshold]
    a = len(sub_rows.index)
    if len(sub_rows.index) == 0:
        sub_prob = 0
    else:
        label = set((sub_rows['class_label'].values)).pop()
        sub_prob = np.sum((sub_rows['class_label'] == label).values, dtype=np.float) / len(sub_rows.index)

    intermidiate_res += (1 - prob_factor) * entropy(sub_prob, (1 - sub_prob))

    return intermidiate_res

def gain_continious(attr, attr_name):
#     attr_name = attr.columns[index_attr]
    attr = attr.sort_values(by=[attr_name])
    label = set((attr['class_label'].values)).pop()
    propability_full_set = np.sum([attr['class_label'] == label], dtype=np.float) / len(attr.index)
    state = attr['class_label'][attr.index[0]]
    gain_dic = {}
    #indexing for each row
    for enum, i in enumerate(attr.index):
        # if i == 8:
        #     pass
        if attr['class_label'][i] != state:
            state = attr['class_label'][i]
            previous_i = attr.index[enum - 1]
            mean = (attr[attr_name][previous_i] + attr[attr_name][i]) / 2.
            gain_dic[i] =  [conditional_entropy(attr, attr_name, mean), mean]
    #find max gain along we got
    index = min(gain_dic.iteritems(), key=operator.itemgetter(1))[0]
    # return [gain, mean] which maximize this attribute

    gain = entropy(propability_full_set, (1 - propability_full_set)) - gain_dic[index][0]
    return_val = (gain, gain_dic[index][1])
    return return_val

def best_threshold(attr):
    max_gain = (-1, None, 0)
    for a_name in attr.columns[:-1]:
        current_gain, threshold = gain_continious(attr, a_name)
        max_gain = max_gain if max_gain[0] >= current_gain else (current_gain, a_name, threshold)
    return max_gain

def TDIDT(attr, tree, node, node_side=None, max_depth=10):
    # attr = attr_orig.copy()
    if node.name != None:
        del attr[node.name]
        # node = tree.find_node(tree.root, node_name)
        node = tree.create_node(node, node_side)
        # node = tree.set_noname_node(node_name, node_side)
    else:
        node = tree.root
    labels = set((attr['class_label'].values))
    if len(labels) == 1:
        node.samples = len(attr.index)
        node.healthy = np.sum(attr['class_label'] == 0)
        node.trisomic = node.samples - node.healthy
        node.label = labels.pop()
        return
    _, attr_name, threshold = best_threshold(attr)
    node.condition = threshold
    node.name = attr_name
    counter = Counter(attr['class_label'].values)
    node.label = counter.most_common(1)[0][0]
    node.samples = len(attr.index)
    node.healthy = np.sum(attr['class_label'] == 0)
    node.trisomic = node.samples - node.healthy
    if node.depth == max_depth:
        return
    TDIDT(attr[attr[node.name] < node.condition], tree, node, 'left', max_depth=max_depth)
    TDIDT(attr[attr[node.name] >= node.condition], tree, node, 'right', max_depth=max_depth)
    # del attr

def check_dependencies(dependencies, root):
    new_root = root + 1
    for key, value in dependencies.items():
        if new_root == key:
            new_root += 1
        if new_root in value:
            new_root = max(value) + 1
    return new_root

def print_node(node, file_name, root, dependencies):
    color = ['#399de506', '#e58139b0']
    if node.left == None and node.right == None:
        with open(file_name, 'a') as f:
            f.write('{} [label=<samples={}<br/>\nhealthy: {}, trisomic: {}<br/>class = {}>, fillcolor="{}"] ;\n'
                    .format(root, node.samples, node.healthy, node.trisomic, node.label, color[int(node.label)]))
        return

    with open(file_name, 'a') as f:
        f.write('{} [label=<{} &le; {}<br/>samples = {}<br/>\nhealthy: {}, trisomic: {}<br/>class = {}>, fillcolor="{}"] ;\n'
                .format(root, node.name, node.condition, node.samples, node.healthy, node.trisomic, node.label, color[int(node.label)]))

    new_root = check_dependencies(dependencies, root)
    dependencies[root].append(new_root)
    print_node(node.left, file_name, new_root, dependencies)

    new_root = check_dependencies(dependencies, root)
    dependencies[root].append(new_root)
    print_node(node.right, file_name, new_root, dependencies)


def write_dependencies(dependencies, file_name):
    with open(file_name, 'a') as f:
        for key, val in dependencies.items():
            for v in val: #4 -> 6 [labeldistance=2.5, labelangle=-45, headlabel="No"] ;
                f.write('{} -> {} ;\n'.format(key, v))

def graphVis(tree, file_name):
    with open(file_name, 'w') as f:
        f.write('digraph Tree {\n'
                'node [shape=box, style="filled", color="black"] ;\n')
        # for ind, depth, name, condition in enumerate(tree.output()):
        #     f.write()
        node = tree.root
    dependencies = defaultdict(list)
    root = 0
    print_node(node, file_name, root, dependencies)
    write_dependencies(dependencies, file_name)

    with open(file_name, 'a') as f:
        f.write('}\n')

def fit(row, node):
    if row[node.name].values[0] < node.condition:
        return node.left
    else:
        return node.right

def classification(data, tree):
    labels = []
    for i in data.index:
        node = tree.root
        while 1:
            node = fit(data.loc[[i]], node)
            if node.left == None and node.right == None:
                labels.append(node.label)
                break
    return np.array(labels)

def accuracy(labels, data):
    print "Test accuracy: ", np.sum(data['class_label'].values == labels, dtype=np.float) / len(labels)


if __name__ == '__main__':
    tree = Tree()
    # df = data
    df = pd.read_csv('./gene_expression_training_orig.csv')
    TDIDT(df, tree, tree.root, max_depth=1)

    df = pd.read_csv('./gene_expression_test_orig.csv')
    labels = classification(df, tree)
    np.savetxt('labels.csv', labels)
    accuracy(labels, df)
    graphVis(tree, 'tr.dot')

