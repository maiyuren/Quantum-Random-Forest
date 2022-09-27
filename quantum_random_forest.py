from pickle import load
from device import set_device
from quantum_decision_tree import QuantumDecisionTreeClassifier
from split_function import SplitFunction, set_kernel_pool, relabel_dataset
from data_construction import check_dataset_relabelled, load_dataset_relabelled, save_dataset_relabelled, all_same
from qsvm import QSVM
import multiprocessing as mp
import numpy as np
import tqdm
import pandas as pd
from sklearn.decomposition import PCA
from collections.abc import Iterable
from kernel_element_storage import KernelElementStorage

MODEL_TYPES = ['clas', 'reg']

PARALLEL = False             # This trains each decision tree in parallel using Pool
NUM_CORES = 3


class QuantumRandomForest:
    """ This object defines the quantum analogue of the Random Forest algorithm. Specifically the model uses
        PQCs for the split functions present in each Decision Tree. """

    def __init__(self, n_qubits, model_type, num_trees, criterion, max_depth=None, min_samples_split=2, tree_split_num=2,
                 ensemble_var='pqc_arch', branch_var='param_rand', dt_type='random', num_classes=None, ensemble_vote_type='ave', num_params_split=None, 
                 num_rand_gen=1, num_rand_meas_q=None, pqc_sample_num=1024, embed=False, svm_num_train=None, nystrom_approx=True,
                 svm_c=None, device='cirq', device_map=None, kernel_pool_filename=None):
        """
        :param n_qubits: number of qubits
        :param model_type: this can either be clas or reg
        :param num_trees: the number of decision trees in the forest
        :param criterion: this the criterion by which we split
        :param max_depth: the maximum depth of each tree
        :param min_samples_split: the minimum number of samples required to split an internal node
        :param tree_split_num: this the number of branches for each tree node
        :param ensemble_var: This is the type of variation occurring in the decision trees
        :param branch_var: This is the variation in the branches - 'param_rand', 'pqc_arch', 'eff_anz_pqc_arch', 'iqp_anz_pqc_arch'. 
                            For QKE you can also select from pool: 'qke_pool'
        :param dt_type: this can either be 'random' or 'qke' referring to the training of the Quantum Decision Tree split function 
        :param num_classes: the number of different classes that we need to identify
        :param ensemble_vote_type: the type of voting that happens across trees --> 'ave' or 'mult'
        :param num_params_split: the number of parameters in the PQC at every split node in decision tree
        :param num_rand_gen: the number of random PQCs generated when choosing for dt_type='random'.
                             For QKE this parameter determines the number of SVMs trained at each node to then choose best.
        :param num_rand_meas_q: the number of different qubits are looked at for best measurement. So 5 looks at a subset of 5 qubits for best measurement. 
                                In QKE this determines the size of the subset of qubits measured
        :param pqc_sample_num: this is the number of times each PQC is sampled
        :param embed: this determines the type of embedding for the input. Default=False, is statevector supplied. 
                      e.g. 'simple' , 'as_params', 'simple_layered', 'as_params_all', 'as_params_iqp
        :param svm_num_train: this is the number of points used to train svm at each node. If a list is given, then we take the element for each depth of the tree
        :param nystrom_approx: True or False that determines whether we perform nystrom approximation. None results in the default value in split_function module.
        :param svm_c: this determines how much the SVM optimisation will avoid misclassification - large C will result in smaller margin with better TRAINING accuracy 
        :param device: 'cirq' is the simulator, or one can provide a IBM device
        :param device_map: this is an array identifying map to real qubits. eg. [4,7,5] => first q is mapped to 4th real qubit
        :param kernel_pool_filename: this chooses the kernel file from which to obtain random kernel PQCs
        """
        assert model_type in MODEL_TYPES, "QFR can be used for either classification(clas) or regression(reg) purposes."
        assert model_type == 'reg' or num_classes is not None, "Number of classes must be specified. "
        assert dt_type != 'qke' or tree_split_num == 2, "Kernel method only possible with a branching of 2."
        assert svm_num_train is None or isinstance(svm_num_train, Iterable) or svm_num_train <= min_samples_split, "Number of samples chosen for SVM must be <= min_samples_split."
        assert not isinstance(svm_num_train, Iterable) or len(svm_num_train) == max_depth - 1, "Supplied 'svm_num_train' must be a list of length (max_depth - 1)." 
        assert not isinstance(branch_var, list) or len(branch_var) == max_depth - 1, "Supplied 'branch_var' must be a list of length (max_depth - 1)." 
        assert not isinstance(embed, list) or len(embed) == max_depth - 1, "Supplied 'embed' must be a list of length (max_depth - 1)." 
        assert not isinstance(num_params_split, list) or len(num_params_split) == max_depth - 1, "Supplied 'num_params_split' must be a list of length (max_depth - 1)." 

        self.n_qubits = n_qubits
        self.model_type = model_type
        self.num_classes = num_classes                                  # Only used in the discrete case
        self.num_trees = num_trees
        self.criterion = criterion
        self.split_num = tree_split_num
        self.tree_max_depth = max_depth
        self.tree_min_samples_split = min_samples_split
        self.dt_type = dt_type
        self.num_rand_gen = num_rand_gen
        self.num_rand_meas_q = n_qubits if num_rand_meas_q is None else num_rand_meas_q 
        self.vote_type = ensemble_vote_type
        self.pqc_sample_num = pqc_sample_num
        self.num_params_split = num_params_split if isinstance(num_params_split, list) else [num_params_split for _ in range(max_depth-1)]
        self.embed = embed if isinstance(embed, list) else [embed for _ in range(max_depth-1)]
        self.branch_var = branch_var if isinstance(branch_var, list) else [branch_var for _ in range(max_depth-1)]
        self.svm_num_train = svm_num_train
        self.nystrom_approx = nystrom_approx
        self.svm_c = svm_c if isinstance(svm_c, list) else [svm_c for _ in range(max_depth-1)]

        self.dev_name = device
        assert device_map is None or len(device_map) == n_qubits, "Device map must align with the {} qubits initiated.".format(n_qubits)
        self.device_map = device_map
        self.kernel_pool_filename = self.set_kernel_pool(kernel_pool_filename)
        self.kernel_el_dict = False

        self.split_fn_list = None 
        self.qdt_list = self.init_ensemble(ensemble_var=ensemble_var)   # List of the decision trees used to form a vote
        self.pca = None
        self.set_device(device, device_map)

        self._all_margins = None    # We stores the margins of the nodes - available only once self.get_all_margins() is called

        # We store recently predicted instances to measure correlation between trees
        self.recent_pred_storage = None

    def __call__(self, instances, return_unc=False, parallel=False, ret_pred_distr=False):
        """ Given a list of instances this function returns a list of outputs voted from the ensemble of trees. """
        assert not parallel or not ret_pred_distr, "To store prediction distributions, testing cannot be run in parallel. This might be required \
                                                    when calculating correlation between trees. "
        if parallel:
            instances_list = split_test_data_for_cores(instances)
            self.remove_sim()
            arguments = [(self, inst) for inst in instances_list]
            with mp.Pool(NUM_CORES) as p:
                results = p.map(_test_pool, arguments)
                self.reset_sim()
                return np.concatenate(results)

        if self.pca is not None:
            instances = self.pca_transform(instances=instances)
        
        if ret_pred_distr:
            self.recent_pred_storage = {i:[] for i in range(self.num_classes)}
        
        if self.dev_name != 'cirq': # For ibm devices we want to send jobs as a group
            all_preds = np.array([qdt(instances, group_call=True) for qdt in self.qdt_list]) # Each column is for each instance
            return np.array([self._vote(all_preds[:, i], return_unc, ret_pred_distr) for i in range(len(instances))])
        else:
            return np.array([self._vote([qdt([instance])[0] for qdt in self.qdt_list], return_unc, ret_pred_distr)
                             for instance in tqdm.tqdm(instances)])

    def train(self, data_df, epochs=1, partition_sample_size=None, pca_dim=None, **kwargs):
        """
        :param data_df: training dataframe with columns X and y
        :param epochs: Epochs can be important as they can reduce sampling error, usually 2 is good
        :param partition_sample_size: determines if data is partitioned through the trees(useful large datasets). Hence
                                      this parameter determines the number randomly sampled for each tree.
        :param pca_dim: this does a PCA dimension reduction to a size of pca_dim
        NOTE: data_df.X must be normalised using 'q_embed_normalise()' in data_construction.py 
        """
        if pca_dim:
            data_df = self.pca_fit_and_transform(pca_dim, data_df)

        data_df = pd.concat([data_df.sample(frac=1).reset_index(drop=True) for _ in range(epochs)])
        assert partition_sample_size is None or isinstance(partition_sample_size,
                                                           int), "partition_sample_size must be an integer."
        if partition_sample_size is not None:
            data_df = self._return_partitioned_data_list(data_df, partition_sample_size)
        else:
            data_df = [data_df] * self.num_trees  # i.e. all trees get the same data

        assert len(data_df) == len(self.qdt_list), "Error in assigning data to trees."

        if PARALLEL:
            self.remove_sim()
            arguments = [(qdt, data_df[i], self.criterion) for i, qdt in enumerate(self.qdt_list)]
            with mp.Pool(NUM_CORES) as p:
                self.qdt_list = p.map(_train_pool, arguments)
            self.reset_sim()  # Being able to run in parallel requires pickle -- cannot have device
        else:
            for i, qdt in enumerate(self.qdt_list):
                print("\n\nTraining tree {} of {} ".format(i + 1, self.num_trees) + "-" * 60)
                if i >= 1:
                    kern_el_dict = self.qdt_list[i - 1].kern_el_dict
                else:
                    kern_el_dict = None
                qdt.train(data_df=data_df[i], kern_el_dict=kern_el_dict)

    def predict(self, instances, return_unc=False, parallel=PARALLEL, calc_tree_corr=False, ret_pred_distr=False):
        ret_pred_distr = True if ret_pred_distr or calc_tree_corr else False
        return self(instances, return_unc, parallel=parallel, ret_pred_distr=calc_tree_corr)

    def test(self, data_df, ret_pred=False, parallel=PARALLEL, calc_tree_corr=False):
        """ Returns accuracy for classification and ME for regression. """
        instances, labels, out = data_df.X, data_df.y, None
        predictions = self(instances, parallel=parallel, ret_pred_distr=calc_tree_corr)
        if self.model_type == 'clas':
            correct = sum(predictions == np.array(labels))
            out = correct/len(labels)
        elif self.model_type == 'reg':
            me = np.sqrt((predictions - np.array(labels)) ** 2)
            out = me.sum() / len(labels)
        if ret_pred:
            return out, predictions
        return out

    def _vote(self, output_list, return_unc=False, ret_pred_distr=False):
        """ Given list of outputs from the decision trees, this implements a voting system to return a model output.
            This voting method is used for predicting a single instance. """
        output_list = remove_None(output_list)      # Some trees in regression make no prediction
        if self.model_type == 'clas':
            return self._vote_class(output_list, ret_pred_distr=ret_pred_distr)
        elif self.model_type == 'reg':
            pass # Todo

    def _vote_class(self, output_list, ret_pred_distr=False):
        pred = {}
        if self.vote_type == 'ave':
            for i in range(self.num_classes):
                tree_preds = [tree_pred[i] for tree_pred in output_list]
                if ret_pred_distr:
                    self.recent_pred_storage[i].append(tree_preds)
                pred[i] = sum(tree_preds) / self.num_trees
        elif self.vote_type == 'mult':
            print("Has not yet been implemented."); exit() # Todo
        elif self.vote_type == 'cont':
            print("Has not yet been implemented."); exit()  # Todo
        else:
            print("No such vote type."); exit()
        
        return max(pred, key=pred.get)

    def init_split_fn(self, ensemble_var):
        # NOTE: ensemble_var becomes useless as the types of split functions are now determined by branch_var 
        if isinstance(self.branch_var, list) or isinstance(self.embed, list):
            assert isinstance(self.branch_var, list), "Given a list of embeddings for each depth, you must also supply a list of brach_var."
            assert isinstance(self.embed, list), "Given a list of branch_var for each depth, you must also supply a list of embedding."
            assert isinstance(self.svm_c, list), "Given a list of branch_var for each depth, you must also supply a list of svm_c."
            assert isinstance(self.num_params_split, list), "Given a list of branch_var for each depth, you must also supply a list of num_params_split."
            assert len(self.branch_var) == len(self.embed), "The number of SF types must match the number of embeddings."
            assert len(self.branch_var) == len(self.svm_c), "The number of SF types must match the number of svm_c."
            assert len(self.branch_var) == len(self.num_params_split), "The number of SF types must match the number of num_params_split."
            sfs = []
            for i, branch_var in enumerate(self.branch_var):
                embed = self.embed[i]
                num_params_split = self.num_params_split[i]
                svm_c = self.svm_c[i]
                assert branch_var != 'iqp_anz_pqc_arch' or embed == 'as_params_iqp', "For 'iqp_anz_pqc_arch' embed must be 'as_params_iqp'."
                sfs.append(self._init_split_fn(branch_var, embed=embed, num_params_split=num_params_split, svm_c=svm_c))
            self.split_fn_list = sfs
            return sfs
        else:
            # This is the old code -- the new code shouldn't go into here and ensemble_var is useless
            print('Code error.'); exit()
            return self._init_split_fn(ensemble_var)
    
    def _init_split_fn(self, sf_type, embed=None, num_params_split=None, svm_c=None):
        embed = self.embed if embed is None else embed
        num_params_split = self.num_params_split if num_params_split is None else num_params_split
        svm_c = self.svm_c if svm_c is None else svm_c
        if self.dt_type == 'qke':
            return self._init_split_fn_qke(sf_type, embed=embed, num_params_split=num_params_split, svm_c=svm_c)
        elif sf_type == 'pqc_arch':
            return SplitFunction.init_rand_pqc(self.n_qubits, num_params_split, self.criterion, self.split_num,
                                               self.pqc_sample_num, embed, sf_type, self.num_rand_meas_q, self.nystrom_approx)
        elif sf_type == 'eff_anz_pqc_arch':
            num_layers = (num_params_split // self.n_qubits) + 1
            return SplitFunction.init_eff_anz_pqc(self.n_qubits, num_layers, self.criterion, self.split_num, 
                                                  self.pqc_sample_num, embed, sf_type, self.num_rand_meas_q, self.nystrom_approx)
        else:
            print("Has not yet been implemented."); exit()

    def _init_split_fn_qke(self, sf_type, embed=None, num_params_split=None, svm_c=None):
        embed = self.embed if embed is None else embed
        num_params_split = self.num_params_split if num_params_split is None else num_params_split
        svm_c = self.svm_c if svm_c is None else svm_c
        assert isinstance(embed, str), "Embedding must be a string."
        assert 'as_params' in embed, "qke has only been implemented for as_params embedding."
        if sf_type == 'pqc_arch':
            return SplitFunction.init_rand_pqc_qke(self.n_qubits, num_params_split, self.criterion, self.split_num,
                                               self.pqc_sample_num, embed, sf_type, self.num_rand_meas_q, self.nystrom_approx, svm_c)
        elif sf_type == 'qke_pool':
            return SplitFunction.init_pqc_qke_pool(self.n_qubits, num_params_split, self.criterion, self.split_num,
                                               self.pqc_sample_num, embed, sf_type, self.num_rand_meas_q, self.nystrom_approx, svm_c)
        elif sf_type == 'eff_anz_pqc_arch':
            # num_layers = (num_params_split // self.n_qubits) + 1
            num_layers = np.ceil(num_params_split / self.n_qubits).astype(int)
            return SplitFunction.init_eff_anz_pqc_qke(self.n_qubits, num_layers, self.criterion, self.split_num, 
                                                  self.pqc_sample_num, embed, sf_type, self.num_rand_meas_q, self.nystrom_approx, svm_c)
        elif sf_type == 'iqp_anz_pqc_arch':
            return SplitFunction.init_iqp_anz_pqc_qke(self.n_qubits, num_params_split, self.criterion, self.split_num,
                                               self.pqc_sample_num, embed, sf_type, self.num_rand_meas_q, self.nystrom_approx, svm_c)
        else:
            print("Has not yet been implemented."); exit()
            
    def init_ensemble(self, ensemble_var='pqc_arch'):
        # This construction of an ensemble must take into account that model_type could be either clas or reg
        if self.model_type == 'clas':
            return [QuantumDecisionTreeClassifier(n_qubits=self.n_qubits, dt_type=self.dt_type,
                                                  split_fn=self.init_split_fn(ensemble_var), # This is what defines the ensemble variation
                                                  num_classes=self.num_classes,
                                                  max_depth=self.tree_max_depth,
                                                  min_samples_split=self.tree_min_samples_split,
                                                  split_num=self.split_num,
                                                  num_rand_gen=self.num_rand_gen, 
                                                  svm_num_train=self.svm_num_train) for _ in range(self.num_trees)]
        elif self.model_type == 'reg':
            return [QuantumDecisionTreeRegressor(n_qubits=self.n_qubits, dt_type=self.dt_type,
                                                 split_fn=self.init_split_fn(ensemble_var), # This is what defines the ensemble variation
                                                 max_depth=self.tree_max_depth,
                                                 min_samples_split=self.tree_min_samples_split,
                                                 split_num=self.split_num,
                                                 num_rand_gen=self.num_rand_gen, 
                                                 svm_num_train=self.svm_num_train) for _ in range(self.num_trees)]

    def pca_fit_and_transform(self, pca_dim, data_df):
        assert len(data_df.X[0]) >= pca_dim, "Dimension of reduction must be less than the original dimension."
        if 'orig_X' in data_df:
            X = data_df.orig_X
        else:
            X = data_df.X
        self.pca = PCA(n_components=pca_dim)
        data_df['orig_X'] = X
        X = self.pca.fit_transform(np.array(list(X.values)))
        data_df['X'] = list(X)
        return data_df

    def pca_transform(self, instances):
        if self.pca is not None:
            return self.pca.transform(np.array(list(instances)))
        else:
            print("PCA not applied.")
            return instances
        
    def relabelling_for_separation(self, train_data_df, test_data_df=None, force_label=False, dataset=None, 
                                    force_save=False, parallel=False, qrf_optimal=False):
        """ This method uses the techniques proposed in 'Power of data in QML' to separate quantum 
            and classical models to obtain quantum advantage. This method will assume all kernels in the 
            forrest of trees are identical and hence pick the first quantum kernel from the first QDT. """
        assert self.dt_type == 'qke', "Relabelling has only been implemented for QKE." 
        assert self.num_classes == 2 or force_label, "Relabelling is only possible with two classes. \
                                                    You can force labelling to two classes by setting force_label=True."
        if dataset is not None and check_dataset_relabelled(dataset, self.num_params_split, self.embed, qrf_optimal=qrf_optimal) and not force_save and not force_label:
            data_df = load_dataset_relabelled(dataset, self.num_params_split, self.embed, qrf_optimal=qrf_optimal)
        else: 
            print("Starting relabelling...\n")
            if test_data_df is not None:
                data_df = pd.concat([train_data_df, test_data_df])
            else:
                data_df = train_data_df

            if all_same(self.embed) and all_same(self.num_params_split):
                print("Relabelling all same. {} data points.".format(len(data_df)))
                sf = self.split_fn_list[0]
                relabel_dataset(dataset=dataset, sf=sf, data_df=data_df, save=True, 
                                parallel=parallel, qrf_optimal=qrf_optimal)
            else:
                d_all = []
                split_data = [data_df.iloc[i: i + np.ceil(len(data_df)/len(self.split_fn_list)).astype(int)] \
                                for i in np.linspace(0, len(data_df), len(self.split_fn_list) +1).astype(int)[:-1]]
                for i, sf in enumerate(self.split_fn_list):
                    print("---- {} of {} ---- (embed={}).".format(i+1, len(self.split_fn_list), sf.embed))
                    assert 'pqc_arch' != sf.branch_var, "Need a fixed kernel throughout forest."
                    dat = split_data[i]
                    d_all.append(relabel_dataset(dataset=dataset, sf=sf, data_df=dat, save=False, qrf_optimal=qrf_optimal))
                
                data_df = pd.concat(d_all)
                save_dataset_relabelled(data_df, dataset=dataset, num_params=self.num_params_split, 
                                        embed=self.embed, qrf_optimal=qrf_optimal)

        if test_data_df is not None:
            return data_df.iloc[:len(train_data_df)], data_df.iloc[len(train_data_df):]
        else:
            return data_df
    
    def return_qsvm(self, nystrom=False, svm_c=None):
        """ This returns the associated QSVM that we can then compare. """
        assert self.dt_type == 'qke', "Associated QSVM has only been implemented for QKE." 
        assert 'pqc_arch' != self.branch_var, "Need a fixed kernel throughout forest."
        if svm_c is None:
            svm_c = self.svm_c
        sf = self.qdt_list[0].split_fn
        return QSVM(sf._ret_svm(nystrom=nystrom), split_fn=sf)

    def load_kernel_el_from_file(self, dataset):
        
        print("When loading kernel elements, only the elements for the first split function is loaded.")
        pqc_sample_num = self.pqc_sample_num
        embed = self.embed[0]
        n_qubits = self.n_qubits
        num_params = self.num_params_split[0]

        kernel_el_dict = KernelElementStorage.generate_from_kernel_matrix_file(dataset, 
                                                        embed=embed, num_params=num_params,
                                                        n_qubits=n_qubits, pqc_sample_num=pqc_sample_num)
        print(kernel_el_dict)
        for qdt in self.qdt_list:
            qdt.kern_el_dict = kernel_el_dict

        print("Kernel Elements successfully loaded into all trees.")
        
    def compute_tree_correlation(self):
        assert self.recent_pred_storage is not None, "No instances tested for computing correlation in trees. "
        corr_dict = {}
        for c in range(self.num_classes):
            pd_p = pd.DataFrame(self.recent_pred_storage[c])
            corr = pd_p.corr(method='spearman')
            corr_dict[c] = np.array(corr)
        return corr_dict
    
    def get_all_margins(self):
        """ Returns a list of margins for each tree. """
        if self._all_margins is None:
            self._all_margins = [qdt.get_margins() for qdt in self.qdt_list]
        return self._all_margins

    def _return_partitioned_data_list(self, data_df, partition_sample_size):
        """ Used for partitioning data into the trees. Often used when we have a large dataset. """
        assert partition_sample_size < len(data_df), "Partition size must be less than the total number of training instances."
        return [data_df.sample(n=partition_sample_size).reset_index(drop=True) \
                for _ in range(self.num_trees)]

    def set_kernel_pool(self, filename):
        """ This method sets the pool from which we randomly choose PQCs at training. """
        if filename is not None:
            set_kernel_pool(filename)
        return filename

    def set_device(self, device, device_map=None):
        set_device(device, device_map)
        self.reset_sim()
        self.dev_name = device
        self.device_map = device_map
        
    def remove_sim(self):
        """ Removes devices so that the QRF can be pickled and saved. """
        [qdt.remove_sim() for qdt in self.qdt_list]
    
    def del_kern_el_dict(self):
        self.kernel_el_dict = None
        [qdt.del_kern_el_dict() for qdt in self.qdt_list]

    def reset_sim(self):
        [qdt.reset_sim(self.criterion) for qdt in self.qdt_list]
    
    def print_trees(self):
        for qdt in self.qdt_list:
            print('\n', '-'*40, '\n')
            qdt.print_tree()

# ----------------------------------------------------------------------------------
# -------------------------- Useful functions --------------------------------------

def set_multiprocessing(b: bool, cores=None):
    global PARALLEL
    global NUM_CORES

    PARALLEL = b
    NUM_CORES = cores if cores is not None else mp.cpu_count()


def split_test_data_for_cores(instances):
    num_inst = -(-len(instances) // NUM_CORES)
    if isinstance(instances, pd.DataFrame):
        return [instances.iloc[i: i + num_inst] for i in range(0, len(instances), num_inst)]
    return [instances[i: i + num_inst] for i in range(0, len(instances), num_inst)]


def _train_pool(argument):
    np.random.seed()
    qdt, data_df, criterion = argument
    return qdt.train(data_df=data_df, criterion=criterion, return_qdt=True)


def _test_pool(argument):
    np.random.seed()
    qrf, data_df = argument
    qrf.reset_sim()
    return qrf(data_df, parallel=False)


def remove_None(l):
    return [i for i in l if i is not None]