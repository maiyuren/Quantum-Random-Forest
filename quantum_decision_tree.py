
from abc import abstractmethod
from data_construction import q_embed_normalise
from split_function import ImpossibleSVMTrainError
import numpy as np
from collections.abc import Iterable
from kernel_element_storage import KernelElementStorage


class QuantumDecisionTree:
    """ This is an abstract method class for the two types of decision trees: classifiers and regressors."""
    def __init__(self, n_qubits, dt_type, split_fn, current_depth=1, max_depth=None,
                 min_samples_split=2, split_num=None, num_rand_gen=None, svm_num_train=None, kern_el_dict=None):

        self.n_qubits = n_qubits
        self.dt_type = dt_type                      # this can be either 'qke' or 'random'
        self.split_num = split_num                  # this is the number of children this node can have
        self.depth = current_depth                  # this is the depth of the current node
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_rand_gen = num_rand_gen            # the number of random PQCs generated at training
        self.svm_num_train = svm_num_train if isinstance(svm_num_train, Iterable) else [svm_num_train for _ in range(max_depth-1)]
        self.split_fn_list = split_fn if isinstance(split_fn, Iterable) else [split_fn for _ in range(max_depth-1)]
        
        assert len(self.split_fn_list) == self.max_depth - 1, "Number of Split Functions supplied must match the depth of the tree."
        self.split_fn = self.split_fn_list[(current_depth - 1) % (max_depth - 1)].copy(split_num, dt_type)    # this creates a new split function for this node, # Note depth starts at 1 while index starts at 0
    
        self.child_dts = []                         # these are the children decision trees
        self._trained_flag = False
        self._leaf = False                          # Determines whether current node is a leaf
        self.leaf_pred_val = None                   # This will have a value regardless of whether it is a leaf
        self.num_data_at_leaf = None
        self.parent = None                          # This is the parent node

        # We store inner products that have been already computed
        self.kern_el_dict = KernelElementStorage() if kern_el_dict is None else kern_el_dict

    def __call__(self, instances, group_call=False):
        """ Given a list of instances, this function returns a list of their labels. """
        assert self._trained_flag, "The decision tree can only be called once it has been trained."
        if self._leaf:
            return self.get_leaf_pred(instances=instances)
        else:
            predictions = []
            if group_call: # This is for sending jobs in groups to the ibm devices
                child_indx_list = self.split_fn(instances, kern_el_dict=self.kern_el_dict)
                preds = [self.child_dts[child_indx]([instances[i] \
                                for i in range(len(instances)) if child_indx==child_indx_list[i]], group_call=True) \
                                for child_indx in range(len(self.child_dts))]       # Note: there are two list-comprehensions 
                return [preds[child_indx].pop(0) for child_indx in child_indx_list]
            for instance in instances:
                child_indx = self.split_fn([instance], kern_el_dict=self.kern_el_dict)[0]
                predictions.append(self.child_dts[child_indx]([instance])[0])

            return predictions

    def train(self, data_df, criterion=None, return_qdt=False, kern_el_dict=None):
        """ This is a recursive method to grow the tree. The data must be given as a DataFrame. """
        print("\n---Training sub-tree of depth:", self.depth, "({} instances)".format(len(data_df)))
        if kern_el_dict is not None:
            print("Kernel Dictionary given to the tree.")
            self.kern_el_dict = kern_el_dict

        if criterion is not None:
            self.reset_sim(criterion)
            # self.split_fn.criterion = criterion
        instances, labels = data_df.X, data_df.y
        assert len(labels) == len(instances), "The number of instances must be the same as the number of labels. "
        K = len(labels)            # The number of instances
        self.init_leaf_pred(data_df)

        if (self.depth < self.max_depth) and (K >= self.min_samples_split):
            try:
                svm_num_train = self.svm_num_train[self.depth - 1]      # Note depth starts at 1 while index starts at 0
                svm_num_train = len(data_df) if svm_num_train is not None and len(data_df) < svm_num_train else svm_num_train # Accounts for RP 
                split_df_list = self.split_fn.train(data_df=data_df, split_type=self.dt_type, num_rand_gen=self.num_rand_gen,
                                                    ret_split_list=True, svm_num_train=svm_num_train, kern_el_dict=self.kern_el_dict)
                self.child_dts = [self._init_and_train_child(split_df) for split_df in split_df_list]
                assert len(self.child_dts) == self.split_num, "Error in splitting children. "
            except ImpossibleSVMTrainError:
                print("All same class.")
                self._leaf = True
        else:
            self._leaf = True

        self._trained_flag = True

        if return_qdt: # Used for training trees in parallel
            self.remove_sim()
            return self

    def _init_and_train_child(self, data_df):

        child = self._init_child()
        child.train(data_df)

        return child

    def get_leaf_pred(self, instances):
        assert self._leaf, "Node must be leaf to be able to get a prediction. "
        return [self.leaf_pred_val for _ in range(len(instances))]

    def print_tree(self, file=None, _prefix="", _last=True):
        """ This function prints the tree of the nodes which have been optimised. """
        print(_prefix, "`- " if _last else "|- ", self._get_tree_value(), sep="", file=file)
        _prefix += "   " if _last else "|  "
        child_count = len(self.child_dts)
        for i, child in enumerate(self.child_dts):
            _last = i == (child_count - 1)
            child.print_tree(file, _prefix, _last)

    def _get_tree_value(self):
        """ This is a subfuntion for the printing of the tree. Returns a string for the identification of that node. """
        return "({}) - {} instances".format({k: round(v, 2) for k,v in self.leaf_pred_val.items()} \
                            if self._leaf else "", self.num_data_at_leaf)
    
    def get_margins(self):
        assert self._trained_flag, "Margins can only be obtained post training."
        margins = []
        if not self._leaf:
            margins.append(self.split_fn.get_margin())
            for qdt in self.child_dts:
                margins = margins + qdt.get_margins()
        
        return margins

    def remove_sim(self):
        self.split_fn.criterion = None
        self.split_fn.pqc.device = None
        # This try statement is to account for the old model code
        try:
            for sf in self.split_fn_list:
                sf.criterion = None
                sf.pqc.device = None
        except AttributeError:
            pass
        [qdt.remove_sim() for qdt in self.child_dts]

    def reset_sim(self, criterion):
        self.split_fn.criterion = criterion
        self.split_fn.pqc.reset_device()
        # This try statement is to account for the old model code
        try:
            for sf in self.split_fn_list:
                sf.criterion = criterion
                sf.pqc.reset_device()
        except AttributeError:
            pass
        [qdt.reset_sim(criterion) for qdt in self.child_dts]
    
    def del_kern_el_dict(self):
        self.kern_el_dict.delete_storage()
        self.kern_el_dict = None 

    @abstractmethod
    def init_leaf_pred(self, data_df):
        pass

    @abstractmethod
    def _init_child(self):
        pass


class QuantumDecisionTreeClassifier(QuantumDecisionTree):
    """ A single decision tree used in ensemble for the random forest for the purpose of classification. """

    def __init__(self, n_qubits, dt_type, split_fn, num_classes, current_depth=1, max_depth=None,
                 min_samples_split=2, split_num=None, num_rand_gen=None, svm_num_train=None, kern_el_dict=None):

        super(QuantumDecisionTreeClassifier, self).__init__(n_qubits, dt_type, split_fn, current_depth, max_depth,
                                                            min_samples_split, split_num, num_rand_gen, svm_num_train, kern_el_dict=kern_el_dict)
        self.num_classes = num_classes

    def init_leaf_pred(self, data_df):
        """ There is no more splitting and the leaf prediction function is initialised. """
        class_embed_list = list(range(self.num_classes))
        tot_instances = len(data_df)
        pred_prob = {}
        self.num_data_at_leaf = len(data_df)
        for embed in class_embed_list:
            num_cl = (data_df.y == embed).sum()
            if tot_instances != 0:
                pred_prob[embed] = (num_cl / tot_instances)  
            else:
                pred_prob[embed] = self.parent.leaf_pred_val[embed]     # We get from the parent node
        self.leaf_pred_val = pred_prob

    def _init_child(self):
        qdt = QuantumDecisionTreeClassifier(self.n_qubits, self.dt_type, self.split_fn_list, self.num_classes,
                                             current_depth=self.depth + 1, max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split, split_num=self.split_num,
                                             num_rand_gen=self.num_rand_gen, svm_num_train=self.svm_num_train, 
                                             kern_el_dict=self.kern_el_dict)
        qdt.parent = self
        return qdt
