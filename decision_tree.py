#!/usr/bin/env python2.7

import pandas as pd
import numpy as np
import cPickle as pickle
import math as mth
import operator
import logging

from argparse import ArgumentParser
from collections import Counter, defaultdict

tree_dump_file = 'tree.txt'
training_file = './gene_expression_training_orig_full.csv'
missed_data_file = 'gene_expression_with_missing_values.csv'
test_file = './gene_expression_test_orig_full.csv'
log_file = './decision_tree.log'
pruned_tree_dot = 'pruned_tree.dot'
pessimistic_pruned_tree_dot = 'pessimistic_pruned_tree.dot'
trisomic = 1
healthy = 0


parser = ArgumentParser(description='assignment3')
parser.add_argument('--build-tree',
                    action='store_true',
                    default=True,
                    help='Build the decision tree'
                    )
parser.add_argument('--prune-tree',
                    action='store_true',
                    default=True,
                    help='Prune decision tree'
                    )
parser.add_argument('--build-rules',
                    action='store_true',
                    default=True,
                    help='Build rules and then prune them'
                    )

logger = logging.getLogger('decision tree logger')
logger.setLevel(logging.DEBUG)
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


def post_traversal(node, use_pessimistic):
    if node.left:
        if not node.left.is_leaf():
            post_traversal(node.left, use_pessimistic)
        else:
            prune(node, use_pessimistic)
            return

    if node.right:
        if not node.right.is_leaf():
            post_traversal(node.right, use_pessimistic)
        else:
            prune(node, use_pessimistic)
            return

    if node.right.is_leaf() and node.left.is_leaf():
        prune(node, use_pessimistic)


def prune(node, use_pessimistic=False):
    if node.right is None:
        logger.error("error no right child")
    if node.left is None:
        logger.error("error no right child")

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

    if use_pessimistic:
        init_err = get_pessimistic_error(1-accuracy_before, total_patterns)
        pruned_err = get_pessimistic_error(1-accuracy_after, total_patterns)
        if init_err >= pruned_err:
            remove_childrens(node)
    else:
        if accuracy_after >= accuracy_before:
            remove_childrens(node)


def remove_childrens(node):
    node.left = None
    node.right = None
    logger.debug('removed nodes left=%s, right=%s from %s' % (node.left,
                                                              node.right,
                                                              node))

def create_rules(tree):
    for n, p in create_path(tree):
        logger.debug('created rules= %s : %s' % (n, p))
    return [Rule(p) for _, p in create_path(tree)]


def create_path(tree):
    for n, p in _create_path(tree.root, []):
        # if n.is_leaf():
        if len(p) > 1:
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
        if self.conjunction_dict == {}:
            logger.error('Rule without conjuctions' % self)

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
            logger.debug('Did not find rule for line = %s, mised value %s' % (i,
                                                                       item['class_label']))
            labels.append(int(not item['class_label']))
            # labels.append(1.0)
            lost_matches += 1
    logger.debug('lost matches in total = %s' % lost_matches)
    return np.array(labels)


def print_rules(rules):
    print('=====================================')
    for r in rules:
        print r


def check_rules(inputs, rules):
    pruned_rules = prune_rules(inputs, rules)
    print('rules before pruning')
    print_rules(rules)
    print('rules after pruning')
    print_rules(pruned_rules)
    print('pruned rules')
    print_rules(set(rules) - set(pruned_rules))
    return pruned_rules

def pessimistic_error_wrapper(inputs, rules):
    labels = rules_classification(inputs, rules)
    e = 1 - accuracy(labels, inputs)
    n = len(labels)
    return get_pessimistic_error(e, n)


def get_pessimistic_error(e, n):
    z = 0.674
    nomitor = (e + z**2/(2*n) + z*mth.sqrt(e/n-e**2/n+z**2/(4*n**2)))
    denominator = 1 + z**2/n
    return nomitor/denominator


def prune_rules(inputs, rules):
    init_err = pessimistic_error_wrapper(inputs, rules)
    pruned_rules = rules[:]
    for i in xrange(len(rules)):
        r = pruned_rules.pop()
        pruned_err = pessimistic_error_wrapper(inputs, pruned_rules)
        if pruned_err < init_err:
            logger.debug('pruned_err %s < init_err %s' % (pruned_err, init_err))
            logger.debug('remove rule %s' % r)
        else:
            logger.debug('pruned_err %s >= init_err %s' % (pruned_err, init_err))
            pruned_rules = [r,] + pruned_rules
    return pruned_rules


#TODO move function to node
def get_label_from_patterns(node):
    return trisomic if node.trisomic_patterns >= node.healthy_patterns else \
        healthy


def get_pruned_tree(tree, use_pessimistic=False):
    post_traversal(tree.root, use_pessimistic)


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
        _clean_test_patterns(args[1].root)
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


def pruned_basycs(df, tree, use_pessimistics=False):
    get_pruned_tree(tree, use_pessimistic=False)
    labels = classification(df, tree)
    acc = accuracy(labels, df)
    msg = "Test accuracy: %s"
    tree_file = pruned_tree_dot
    if use_pessimistics:
        tree_file = pessimistic_pruned_tree_dot
        msg += ' (using pessimistic error)'
    graphVis(t_pruned_by_heuristic, tree_file)
    print msg % acc


def test_random(tree=None, rules=None):
    # cycles
    n = 10
    accurancies = []
    for i in xrange(n):
        df = pd.read_csv(missed_data_file)
        df[np.isnan(df)] = np.random.random((df.shape))
        if tree:
            labels = classification(df, tree)
            accurancies.append(accuracy(labels, df))
        if rules:
            accurancies.append(1-pessimistic_error_wrapper(df, rules))
    return sum(accurancies)/float(n)


def test_set_for_tree(tree):
    accuracy_dict = {}
    df = pd.read_csv(missed_data_file)
    df_median = df.fillna(df.median())
    df_mean = df.fillna(df.mean())
    accuracy_dict['median'] = accuracy(classification(df_median, tree), df_median)
    accuracy_dict['mean'] = accuracy(classification(df_mean, tree), df_mean)
    accuracy_dict['random'] = test_random(tree=tree)
    return accuracy_dict

def test_set_for_rules(rules):
    accuracy_dict = {}
    df = pd.read_csv(missed_data_file)
    df_median = df.fillna(df.median())
    df_mean = df.fillna(df.mean())
    accuracy_dict['median'] = 1-pessimistic_error_wrapper(df_median, rules)
    accuracy_dict['mean'] = 1-pessimistic_error_wrapper(df_mean, rules)
    accuracy_dict['random'] = test_random(rules=rules)
    return accuracy_dict


if __name__ == '__main__':
    args = parser.parse_args()
    if args.build_tree:
        tree = Tree()
        df = pd.read_csv(training_file)
        TDIDT(df, tree, tree.root, max_depth=3)

        df = pd.read_csv(test_file)
        labels = classification(df, tree)
        np.savetxt('labels.csv', labels)
        acc = accuracy(labels, df)
        print "Test accuracy: %s" % acc
        graphVis(tree, 'tr.dot')
        pickle.dump(tree, open(tree_dump_file, 'w'))

    if args.prune_tree:
        df = pd.read_csv(test_file)
        t_pruned_by_heuristic = pickle.load(open(tree_dump_file, 'r'))
        pruned_basycs(df, t_pruned_by_heuristic)

        t_pruned_by_pessimistic = pickle.load(open(tree_dump_file, 'r'))
        pruned_basycs(df, t_pruned_by_pessimistic, use_pessimistics=True)

        tree = pickle.load(open(tree_dump_file, 'r'))
        print('Init tree accuracies %s' % test_set_for_tree(tree))
        print('Heuristic pruned accuracies %s' %
              test_set_for_tree(t_pruned_by_heuristic))
        print('Pessimistic pruned accuraices %s' %
              test_set_for_tree(t_pruned_by_pessimistic))

    if args.build_rules:
        df = pd.read_csv(training_file)
        tree = pickle.load(open(tree_dump_file, 'r'))
        rules = create_rules(tree)
        pruned_rules = check_rules(df, rules)
        init_err = pessimistic_error_wrapper(df, rules)
        pruned_err = pessimistic_error_wrapper(df, pruned_rules)
        print('rules accuracy %s (before pruning)' % (1-init_err))
        print('rules accuracy %s (after pruning)' % (1-pruned_err))
        print('Rules accuraices %s' %
              test_set_for_rules(pruned_rules))

