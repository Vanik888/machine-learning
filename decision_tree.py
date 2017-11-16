#!/usr/bin/env python2.7

import pandas as pd
import numpy as np
import cPickle as pickle
import math as mth
import operator
import logging

from collections import Counter, defaultdict

tree_dump_file = 'tree.txt'
training_file = './gene_expression_training_orig_full.csv'
test_file = './gene_expression_test_orig_full.csv'
log_file = './decision_tree.log'
trisomic = 1
healthy = 0


logger = logging.getLogger('decision tree logger')
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


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
        self.healthy_patterns = 0
        self.trisomic_patterns = 0

    def is_leaf(self):
        return self.left is None and self.right is None

    def __str__(self):
        return 'node %s, healthy: %s, trisomic: %s, left: %s, right: %s' % (
            self.name,
            self.healthy_patterns,
            self.trisomic_patterns,
            self.left,
            self.right)

    def __repr__(self):
        return 'name: %s' % self.name


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


def post_traversal(node):
    if node.left:
        if not node.left.is_leaf():
            post_traversal(node.left)
        else:
            prune(node)
            return

    if node.right:
        if not node.right.is_leaf():
            post_traversal(node.right)
        else:
            prune(node)
            return

    if node.right.is_leaf() and node.left.is_leaf():
        prune(node)


def prune(node):
    if node.right is None:
        print "error no right child"
    if node.left is None:
        print "error no left child"

    left_label = get_label_from_patterns(node.left)
    right_label = get_label_from_patterns(node.right)
    parent_label = get_label_from_patterns(node)
    if left_label == right_label:
        remove_childrens(node)
        return
    results_list = {'left': (node.left.healthy_patterns,
                             node.left.trisomic_patterns),
                    'right': (node.right.healthy_patterns,
                              node.right.trisomic_patterns),
                    'parent': (node.healthy_patterns, node.trisomic_patterns)}
    total_patterns = float(node.healthy_patterns + node.trisomic_patterns)
    accuracy_before = (results_list['left'][left_label] +
                      results_list['right'][right_label]) / total_patterns
    accuracy_after = results_list['parent'][parent_label] / total_patterns
    if accuracy_after >= accuracy_before:
        remove_childrens(node)


def remove_childrens(node):
    node.left = None
    node.right = None
    print 'removed nodes left=%s, right=%s from %s' % (node.left,
                                                      node.right,
                                                      node)


def create_rules(tree):
    for n, p in create_path(tree):
        print n, p
    return [Rule(p) for _, p in create_path(tree)]


def create_path(tree):
    for n, p in _create_path(tree.root, []):
        # if n.is_leaf():
            yield n, p


def _create_path(node, path):
    updated_path = path[:]
    updated_path.append(node)
    yield node, updated_path
    if node.left:
        for lc in _create_path(node.left, updated_path):
            yield lc
    if node.right:
        for rc in _create_path(node.right, updated_path):
            yield rc


class Conjunction:
    def __init__(self, condition, sign):
        self.sign = sign
        self.condition = condition

    def compare_to_rule(self, val):
        if self.sign == '<':
            return val < self.condition
        else:
            return val >= self.condition

    def __str__(self):
        return 'val %s %s' % (self.sign, self.condition)

    def __repr__(self):
        return self.__str__()


class Rule:
    def __init__(self, nodes):
        self.result = nodes[-1:][0].label
        self.conjunction_dict = {}
        self.set_conjunctions(nodes)

    def set_conjunctions(self, nodes):
        for i in xrange(len(nodes)-1):
            sign = '<' if nodes[i].left == nodes[i+1] else '>='
            conjunction = Conjunction(nodes[i].condition, sign)
            self.conjunction_dict[nodes[i].name] = conjunction

    def get_result(self):
        return self.result

    def check_match(self, input):
        rule_matched = False
        for k, v in self.conjunction_dict.items():
            rule_matched = v.compare_to_rule(input[k])
            if not rule_matched:
                return rule_matched
        return rule_matched

    def __str__(self):
        conj_str = ['%s: %s' % (k, v) for k, v in self.conjunction_dict.items()]
        return ' and '.join(conj_str) + ',re = %s' % self.result

def find_match(input, rules):
    for r in rules:
        if r.check_match(input):
            return r
    return None

def rules_classification(inputs, rules):
    labels = []
    lost_matches = 0
    for i in xrange(len(inputs.index)):
        item = inputs.loc[i]
        matched_rule = find_match(item, rules)
        if matched_rule:
            labels.append(matched_rule.get_result())
        else:
            print('Did not find rule for line = %s, mised value %s' % (i,
                                                                       item['class_label']))
            # labels.append(int(not item['class_label']))
            labels.append(1.0)
            lost_matches += 1
    print('lost matches in total = %s' % lost_matches)
    return np.array(labels)


def print_rules(rules):
    for r in rules:
        print r

def check_rules(inputs, tree):
    rules = create_rules(tree)
    pruned_rules = prune_rules(inputs, rules)
    print('before pruning')
    print_rules(rules)
    print('after pruning')
    print_rules(pruned_rules)
    print('pruned rules')
    print_rules(set(rules) - set(pruned_rules))



def get_pessimistic_error(inputs, rules):
    labels = rules_classification(inputs, rules)
    e = 1 - accuracy(labels, inputs)
    z = 0.674
    n = len(labels)
    nomitor = (e + z**2/(2*n) + z*mth.sqrt(e/n-e**2/n+z**2/(4*n**2)))
    denominator = 1 + z**2/n
    return nomitor/denominator


def prune_rules(inputs, rules):
    init_err = get_pessimistic_error(inputs, rules)
    print('rules pesimistic error %s (before pruning)' % init_err)
    pruned_rules = rules[:]
    for i in xrange(len(rules)):
        r = pruned_rules.pop()
        pruned_err = get_pessimistic_error(inputs, pruned_rules)
        if pruned_err < init_err:
            print('pruned_err %s < init_err %s' % (pruned_err, init_err))
            print('remove rule %s' % r)
        else:
            print('pruned_err %s >= init_err %s' % (pruned_err, init_err))
            pruned_rules = [r,] + pruned_rules
    return pruned_rules




#TODO move function to node
def get_label_from_patterns(node):
    return trisomic if node.trisomic_patterns >= node.healthy_patterns else \
        healthy

def post_traversal_wrapper(tree):
    post_traversal(tree.root)


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
            f.write('{} [label=<samples={}<br/>\n'
                    'healthy: {}, trisomic: {}<br/>\n'
                    'tested healthy: {}, tested trisomatic: {}<br/>\n '
                    'class= {}>, fillcolor="{}"] ;\n'
                    .format(root, node.samples,
                            node.healthy, node.trisomic,
                            node.healthy_patterns, node.trisomic_patterns,
                            node.label, color[int(node.label)]))
        return

    with open(file_name, 'a') as f:
        f.write('{} [label=<{} &le; {}<br/>samples = {}<br/>\n'
                'healthy: {}, trisomic: {}<br/>\n'
                'tested healthy: {}, tested trisomatic: {}<br/>\n'
                'class = {}>, fillcolor="{}"] ;\n'
                .format(root, node.name, node.condition, node.samples,
                        node.healthy, node.trisomic,
                        node.healthy_patterns, node.trisomic_patterns,
                        node.label, color[int(node.label)]))

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


def clean_test_patterns(func):
    def wrapper(*args):
        _clean_test_patterns(tree.root)
        return func(*args)
    return wrapper

def _clean_test_patterns(node):
    if node.left is not None:
        _clean_test_patterns(node.left)
    if node.right is not None:
        _clean_test_patterns(node.right)
    node.healthy_patterns = 0
    node.trisomic_patterns = 0


@clean_test_patterns
def classification(data, tree, count_patterns=False):
    labels = []
    for i in data.index:
        node = tree.root

        while 1:
            node = fit(data.loc[[i]], node)
            if data.loc[i]['class_label'] == trisomic:
                node.trisomic_patterns += 1
            else:
                node.healthy_patterns += 1
            if node.left == None and node.right == None:
                labels.append(node.label)
                break
    return np.array(labels)

def accuracy(labels, data):
    acc = np.sum(data['class_label'].values == labels, dtype=np.float) / len(labels)
    return acc


if __name__ == '__main__':
    do_not_rebuild_tree = True
    if not do_not_rebuild_tree:
        tree = Tree()
        df = pd.read_csv(training_file)
        TDIDT(df, tree, tree.root, max_depth=6)

        df = pd.read_csv(test_file)
        labels = classification(df, tree)
        np.savetxt('labels.csv', labels)
        acc = accuracy(labels, df)
        print "Test accuracy: %s" % acc
        graphVis(tree, 'tr.dot')

        pickle.dump(tree, open(tree_dump_file, 'w'))
    else:
        tree = pickle.load(open(tree_dump_file, 'r'))
        post_traversal_wrapper(tree)
        df = pd.read_csv(test_file)
        labels = classification(df, tree)
        acc = accuracy(labels, df)
        print "Test accuracy: %s" % acc
        graphVis(tree, 'pruned_tree.dot')

        tree = pickle.load(open(tree_dump_file, 'r'))
        create_rules(tree)
        check_rules(df, tree)



