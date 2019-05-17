import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('clean_dataset.txt')
# --------------------------------------------------- #
#     Change path below to load your test dataset     #
#                                                     #
path = 'noisy_dataset.txt'  #
#                                                     #
#     The methods that are executed can be found      #
#            at the bottom of the file                #
# --------------------------------------------------- #

noisy_dataset = np.loadtxt(path)


# ----------------- Utils functions ----------------- #


def split_dataset(dataset, idx):
    return dataset[:idx, :], dataset[idx:, :]


# Extract a part (from start_index and of len lenght) and returns the extract and the remaining array
def extract_from_array(full_array, start_index, length):
    left_part, tmp = split_dataset(full_array, start_index)
    extract, right_part = split_dataset(tmp, length)
    remaining = np.concatenate((left_part, right_part))
    return extract, remaining


# Compute the entropy of a dataset : will be usefull in the split function
def compute_entropy(dataset):
    unique, counts = np.unique(dataset[:, 7], return_counts=True)
    entropy = 0
    n = dataset[:, 7].shape[0]
    for id, label in enumerate(unique):
        probability = counts[id] / n
        entropy -= probability * np.log2(probability)
    return entropy


# ---------------- The Split function ----------------- #


# Here, we design a function find_split, which chooses the attribute and the value that results in the highest
# information gain when splitting a dataset on a node of the tree.
def find_split(full_dataset):
    # Start by evaluating the entropy of the full dataset
    entropy_full = compute_entropy(full_dataset)
    n = full_dataset.shape[0]
    best_feature, best_value, best_gain = 0, 0, 0
    # We will evaluate every possible split in each of the 7 features
    for feature in range(7):
        # We start by sorting the dataset in ascending order on the current feature
        sorted_dataset = full_dataset[full_dataset[:, feature].argsort()]
        minval, maxval = sorted_dataset[0][feature], sorted_dataset[-1][feature]
        # We discard the features containing the same value for all samples (can lead to wrong splits)
        if minval == maxval:
            continue
        for idx in range(len(sorted_dataset[:, feature])):
            # Split the data on the current index
            data_left, data_right = split_dataset(sorted_dataset, idx)
            # Evaluate the entropy of each sub dataset
            entropy_left, entropy_right = compute_entropy(data_left), compute_entropy(data_right)
            # Compute the resulting gain of such a split
            gain = entropy_full - ((data_left.shape[0] / n) * entropy_left + (data_right.shape[0] / n) * entropy_right)
            # If this is the best split so far, we keep track of its characteristics
            if gain > best_gain:
                best_feature = feature
                best_gain = gain
                if sorted_dataset[idx, feature] == minval:
                    # Avoid the case where the split point is at the edge of the dataset (and thus we could not split)
                    best_value = sorted_dataset[idx, feature] + 0.5
                elif sorted_dataset[idx, feature] == maxval:
                    best_value = sorted_dataset[idx, feature] - 0.5
                else:
                    best_value = sorted_dataset[idx, feature]
    return best_feature, best_value


# ---------------- Build a tree from dataset ----------------- #

# The "training" algorithm: building of the decision tree from the dataset
def decision_tree_learning(data, depth):
    # If there's only one label on the current dataset, we have found a leaf of the tree
    if len(np.unique(data[:, 7])) == 1:
        terminal_node = {'attr': 'leaf', 'value': data[0, 7], 'left': {}, 'right': {}}
        return terminal_node, depth
    else:
        attr, val = find_split(data)
        # Get a "mask" (array of booleans which are the results of the arithmetic test on the array elements)
        mask_less = data[:, attr] <= val
        mask_more = data[:, attr] > val
        l_dataset = data[mask_less]
        r_dataset = data[mask_more]
        # Recursively build the tree
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
        node = {'attr': attr, 'value': val, 'left': l_branch, 'right': r_branch}
        return node, max(l_depth, r_depth)


# ---------------- Bonus : Graphics functions ----------------- #

def plot_graph_dot(tree, x=0, y=0, depth=0, max_depth=None):
    # Parameters (spaces between nodes, and box format)
    bbox_parameters = dict(boxstyle="round,pad=0.3", fc="white", ec="black")
    delta_x = 1 / (2 ** depth)
    delta_y = 1
    plt.axis('off')

    # Bubble text for leafs
    if tree['attr'] == 'leaf':
        plt.text(x, y, 'leaf:' + str(tree['value']), ha="center", va="center", bbox=bbox_parameters)

    # Plot a segment between the nodes and its children, and apply recursively the function to the children
    else:
        plt.text(x, y, '[X' + str(tree['attr']) + ' < ' + str(tree['value']) + ']', ha="center", va="center",
                 bbox=bbox_parameters)

        # Check if depth below max_def before doing recursion
        if max_depth == None or (max_depth != None and depth < max_depth):
            plt.plot([x, x - delta_x], [y, y - delta_y], color='black')
            plt.plot([x, x + delta_x], [y, y - delta_y], color='black')
            plot_graph_dot(tree['left'], x - delta_x, y - delta_y, depth + 1, max_depth)
            plot_graph_dot(tree['right'], x + delta_x, y - delta_y, depth + 1, max_depth)


def plot_graph(tree, max_depth=None):
    # Function to plot a tree. A parameter max_depth can be entered
    if not max_depth is None:
        actual_depth = min(max_depth, tree[1])
    else:
        actual_depth = tree[1]
    # Auto adujst the size of the graph for low depth trees
    if actual_depth < 5:
        plt.figure(1, figsize=(1.5 * 2 ** actual_depth, 2 * actual_depth))
    else:
        plt.figure(1, figsize=(60, 20))
    plot_graph_dot(tree[0], max_depth=actual_depth)

    file_name = 'tree_' + str(actual_depth) + '.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')


# ---------------- Evaluation and metrics ----------------- #

# Confusion matrix for labels 1, 2, 3, 4.
def confusion_matrix(true_labels, predicted_labels):
    cm = np.zeros((4, 4))
    for c in range(len(true_labels)):
        i, j = int(true_labels[c]), int(predicted_labels[c])
        cm[i - 1, j - 1] += 1
    return cm


# We design an evaluation method that will take as inputs a test database and a trained_tree and outputs the label
# predictions of the model, the true label of the test (will be used to build the confusion matrix) as well as the
# accuracy of the model.
def evaluate(test_db, trained_tree):
    test_labels = test_db[:, 7]
    test_predictions = np.zeros(len(test_labels))
    for idx, sample in enumerate(test_db):
        node = trained_tree
        leaf = False
        while not leaf:
            attr, val, l_branch, r_branch = node['attr'], node['value'], node['left'], node['right']
            if l_branch == {} and r_branch == {}:
                leaf = True
            else:
                if sample[attr] <= val:
                    node = l_branch
                elif sample[attr] > val:
                    node = r_branch
                else:
                    print("ERROR, can't decide which branch to follow")
        test_predictions[idx] = node['value']
    error = float((((test_labels - test_predictions) != 0).sum()) / len(test_labels))
    accuracy = 1 - error
    return accuracy, test_predictions, test_labels


# Takes the confusion matrix as an input and outputs the corresponding recall and precision for each class
def recall_precision(cm):
    nb_classes = cm.shape[0]
    recall = np.zeros(nb_classes)
    precision = np.zeros(nb_classes)
    for idx in range(nb_classes):
        if (idx + 1) < nb_classes:
            false_negative = np.concatenate((cm[idx, :idx], cm[idx, (idx + 1):]))
            false_positive = np.concatenate((cm[:idx, idx], cm[(idx + 1):, idx]))
        else:
            false_negative = cm[idx, :idx]
            false_positive = cm[:idx, idx]
        true_positive = cm[idx, idx]
        recall[idx] = true_positive / (true_positive + false_negative.sum())
        precision[idx] = true_positive / (true_positive + false_positive.sum())
    return recall, precision


# Outputs F1-measure by combining Recall and Precision
def f1_measure(recall, precision):
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


# Let's now define a small function that will digest the exhaustive results from cross-validation
def analyse_metrics(cv_results, folds):
    accuracies, recall, precision, f1 = np.zeros(folds), np.zeros((folds, 4)), np.zeros((folds, 4)), np.zeros(
        (folds, 4))
    sum_cm = np.zeros((4, 4))
    for i in range(folds):
        sum_cm += cv_results[i][1]  # Element wise sum of all confusion matrices
        accuracies[i], recall[i], precision[i], f1[i] = cv_results[i][0], cv_results[i][2], cv_results[i][3], \
                                                        cv_results[i][4]
    mean_accuracy = accuracies.mean()
    dataset_size = sum_cm.sum()
    mean_cm = sum_cm.astype(float) / dataset_size
    mean_recall, mean_precision, mean_f1 = recall.mean(0), precision.mean(0), f1.mean(0)
    return mean_accuracy, mean_cm, mean_recall, mean_precision, mean_f1, sum_cm


# ---------------- 10-fold cross-validation ----------------- #

# This algorithm does the following:
# - Shuffles the dataset and split it into 10 folds (tuples of the form: (training set, validation set))
# - For each fold:
#     - build the tree on the base of the training dataset
#     - evaluate de performance of the model on the validation dataset, by outputting the following metrics:
#       the accuracy, the confusion matrix, the recall, precision and f1-measure.
# Warning : In this version, there is no differentiation between validation and test sets, because we do not
# tune any parameter based on the result of the validation dataset : the validation dataset is actually a test
# dataset
def cross_validation(dataset, folds):
    # Shuffle the dataset
    np.random.shuffle(dataset)
    set_size = int(dataset.shape[0] / folds)  # Warning: choose a fold number which divides the dataset length
    # Creating tuples of (training,testing) folds
    sets = []
    for fold in range(folds):
        testing, training = extract_from_array(dataset, fold * set_size, set_size)
        sets += [(training, testing)]
    # Initialises the array that will contain the metrics of each fold
    results = [0] * folds
    # cm_matrices = np.zeros((4, 4))
    for idx, fold in enumerate(sets):
        trained_tree = decision_tree_learning(fold[0], 0)[0]
        evaluation = evaluate(fold[1], trained_tree)
        accuracy = evaluation[0]
        cm = confusion_matrix(evaluation[2], evaluation[1])
        recall, precision = recall_precision(cm)
        f1 = f1_measure(recall, precision)
        metrics = [accuracy, cm, recall, precision, f1]
        results[idx] = metrics
    return results


# ------------------------ Pruning ------------------------ #
# For the pruning, we will need the following functions to execute the traversal of the decision tree.

# Creates a new tree with a pair of leaves merged according to a path
def change_node(node, path, value):
    # Last node
    if len(path) == 1:
        return {'attr': 'leaf', 'value': value, 'left': {}, 'right': {}}

    # Other nodes
    else:
        direction = path.pop()[2]
        if direction == 'left':
            return {'attr': node['attr'], 'value': node['value'],
                    'left': change_node(node[direction], path, value), 'right': node['right']}
        else:
            return {'attr': node['attr'], 'value': node['value'],
                    'left': node['left'], 'right': change_node(node[direction], path, value)}


def merge_some_leaves(tree, rev_path, value):
    new_tree = tree.copy()
    path = rev_path[::-1]  # Reversed path
    new_tree = change_node(new_tree, path, value)
    return new_tree


# Compares the performance of two trees on the test dataset and returns the best tree
def compare_trees(test_db, original_tree, new_tree):
    error_init = 1 - evaluate(test_db, original_tree)[0]  # error = 1 - accuracy
    error = 1 - evaluate(test_db, new_tree)[0]
    # Compare the "new" error with the initial one
    if error < error_init:
        return new_tree
    else:
        return original_tree


# Returns a pruned tree according to the test dataset
def pruning(test_db, tree):
    whole_tree_copy = tree.copy()
    node, path, prun_tree = pruning_aux(whole_tree_copy, [], whole_tree_copy, test_db)
    return prun_tree


def pruning_aux(node, path, whole_tree, test_data):
    # Leaf
    if node['attr'] == 'leaf':
        return node, path, whole_tree

    # Node
    else:
        path += [(node['attr'], node['value'], 'left')]

        updated_node = node.copy()
        updated_tree = whole_tree.copy()
        updated_node['left'], path, updated_tree = pruning_aux(updated_node['left'], path, updated_tree, test_data)
        path.pop()
        path += [(node['attr'], node['value'], 'right')]

        updated_node['right'], path, updated_tree = pruning_aux(updated_node['right'], path, updated_tree, test_data)

        # If there are two leaves under the current node
        if (updated_node['left']['attr'] == 'leaf') and (updated_node['right']['attr'] == 'leaf'):
            labels = [updated_node['left']['value'], updated_node['right']['value']]
            new_tree1 = merge_some_leaves(updated_tree, path, labels[0])
            new_tree2 = merge_some_leaves(updated_tree, path, labels[1])
            tree1 = compare_trees(test_data, updated_tree, new_tree1)
            tree2 = compare_trees(test_data, updated_tree, new_tree2)
            tree = compare_trees(test_data, tree1, tree2)

            # If merging the two leaves is better for the accuracy
            if tree != updated_tree:
                if tree == tree1:
                    label = labels[0]
                else:
                    label = labels[1]

                updated_node = {'attr': 'leaf', 'value': label, 'left': {}, 'right': {}}
                updated_tree = tree.copy()

        path.pop()
        return updated_node, path, updated_tree


# ------------------------ 10-fold cross validation : version 2 ------------------------ #


def separate_dataset(dataset, folds):
    output = []
    np.random.shuffle(dataset)
    set_size = int(dataset.shape[0] / folds)  # Warning: choose a fold number which divides the dataset length
    for fold in range(folds):
        test_data, train_val = extract_from_array(dataset, fold * set_size, set_size)
        sub_fold_data = []
        for sub_fold in range(folds - 1):
            validation_data, training_data = extract_from_array(train_val, sub_fold * set_size, set_size)
            sub_fold_data.append([training_data, validation_data])
        fold_data = [test_data, sub_fold_data]
        output.append(fold_data)
    return output


# Inputs : dataset to train and evaluate decision_tree on
# Inputs : number of folds (10 for 10-folds cross validation)
# Inputs : all_combination : True if train and test on all possible combinations (i.e 90 trees for 10 fold cross)
#                            False if split the dataset only once (i.e in 1 test dataset and 9 trees for 10 fold)
# Outputs: Matrix of accuracies with and without pruning on the format (Fold x Sub_Fold x 2)
def cross_validation_pruning(dataset, folds, all_combination=True):
    # Creating lists of (training, validation, testing) folds
    test_train_val = separate_dataset(dataset, folds)
    # Prepare the matrix that will contain the results
    if all_combination:
        results = np.zeros((folds, folds - 1, 2))
    else:
        results = np.zeros((folds - 1, 2))
    # cm_matrices = np.zeros((4, 4))

    for idx, fold in enumerate(test_train_val):
        if not all_combination and idx != 0:
            continue
        test_data = fold[0]
        training_validation = fold[1]
        for sub_idx, sub_fold in enumerate(training_validation):
            training_data = sub_fold[0]
            validation_data = sub_fold[1]
            # Now that the 3 dataset are clearly separated, let's train the tree
            trained_tree = decision_tree_learning(training_data, 0)[0]
            # Now let's prune it using the validation dataset to tune the pruning
            pruned_tree = pruning(validation_data, trained_tree)
            # We evaluate the accuracy before and after pruning
            acc_noprun, acc_prun = evaluate(test_data, trained_tree)[0], evaluate(test_data, pruned_tree)[0]
            # Store the results
            if all_combination:
                results[idx, sub_idx] = np.array([acc_noprun, acc_prun])
            else:
                results[sub_idx] = np.array([acc_noprun, acc_prun])
            print("Done with Test ", idx, ".", sub_idx)
        print("--------- Done with fold #", idx)

    return results


#################################################
#                                               #
#   Find below the methods that are executed    #
#                                               #
#################################################

# Indicate if you have time to run the program (False: appr. 2 min, True: appr. 20 min)
got_time = False
print("10-fold cross validation on the dataset ", path, " starts")
print("The parameter 'got_time' is currently set to: ", got_time)

results = cross_validation_pruning(noisy_dataset, 10, all_combination=got_time)
print("The mean accuracy before pruning is :")
if not got_time:
    print(results.mean(0)[0])
if got_time:
    print(results.mean(0).mean(0)[0])
print("The mean accuracy after pruning is :")
if not got_time:
    print(results.mean(0)[1])
if got_time:
    print(results.mean(0).mean(0)[1])

tree = decision_tree_learning(noisy_dataset, 0)

print("Graphical representation in progress (full depth)")
plot_graph(tree)
print("File successfully exported in the current folder")

print("Graphical representation in progress (max_depth=4)")
plot_graph(tree, 4)
print("File successfully exported in the current folder")