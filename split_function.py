
from multiprocessing import Pool
import numpy as np
from pickle_functions import unpickle_file
from pqc import PQC, exp_val_from_counts_list
import itertools
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
pd.options.mode.chained_assignment = None
from pqc import set_kernel_pool
import os
import more_itertools as mit
from sklearn.gaussian_process.kernels import RBF
from nystrom import Nystrom
from data_construction import check_dataset_relabelled, get_stilted_dataset, save_dataset_relabelled, relabelled_filename, save_kernel_for_dataset_relabelled, all_datasets
from sklearn.model_selection import cross_val_score


EPSILON = 0.50                  # Part of the definition of defining the delicacy of a PQC. 0.25 is for p(1-p) at p=1/2 and *2 for 2std 
DELICACY_PROP = 0.3             # If more than DELICACY_PROP proportion of the training set results are delicate, then
                                # the PQC is run again with a different architecture

SVM_KERNEL_SYM = False          # This determines wheter the kernel is symmeterised
ENFORCE_PSD = False
NYSTROM = False
NYSTROM_CLASSICAL_KERNEL = 'linear'
CLASS_SELECTION = 'split' # "OAA" or "split"

class SplitFunction:
    """ This object contains the function used to split the set of quantum states into the lower branches. Each
        node in the decision tree would contain this parameterised SplitFunction that can be learned. """
    def __init__(self, criterion, split_num, pqc, pqc_sample_num, embed=False, branch_var='param_rand', 
                 num_rand_meas_q=None, nystrom=None, svm_c=None):
        assert isinstance(pqc, PQC), "pqc must be of object PQC."
        assert isinstance(embed, str), "Embedding not a string."
        assert isinstance(branch_var, str), "branch_var not a string."
        
        self.criterion = criterion
        self.split_num = split_num
        self.pqc_sample_num = pqc_sample_num
        self.embed = embed
        self.branch_var = branch_var
        self.num_rand_meas_q = num_rand_meas_q

        self.pqc = pqc  
        # self.pqc = pqc.gen_rand(1, param_init_type=branch_var)[0] # if you need this, then make sure that it is compatible with QKE
        self._trained_flag = False
        self._epsilon = EPSILON / np.sqrt(pqc_sample_num if not isinstance(pqc_sample_num, str) else 1) # Used for checking delicacy of PQC

        self.split_type = None          # This is defined at training
        self._svm = None
        self.svm_num_train = None
        self.svm_c = 1.0 if svm_c is None else svm_c
        self._svm_sym = SVM_KERNEL_SYM #if num_rand_meas_q != pqc.n_qubits else False # You don't need to symmeterise kernel if measuring all qubits
        self.nystrom = NYSTROM if nystrom is None else nystrom
        self.feat_map_nystrom = None
        self._train_kernel = None

        # QSVM
        self.kern_el_dict_for_qsvm = None
        self._flag_qsvm = False

    def __call__(self, instances, pqc=None, _training=False, _pqc_out_list=None, kern_el_dict=None):
        """ This function receives a list of instances and returns a child index for each instance. """
        assert _pqc_out_list is None or len(_pqc_out_list) == len(instances), "Error in provided _pqc_out_list."
        assert _pqc_out_list is None or self.split_type=='random', "_pqc_out_list can only be used with 'random' split function."
        if len(instances) == 0:
            return []
        
        pqc = self.pqc if pqc is None else pqc

        if self.split_type == 'random':
            pqc_out_list = pqc(instances) if _pqc_out_list is None else _pqc_out_list
            out_indices = [self.map_pqc_out(pqc_out, _training=_training) for pqc_out in pqc_out_list]
        elif self.split_type == 'qke':
            out_indices = self.svm_predict(instances, kern_el_dict=kern_el_dict)

        if not _training:
            return out_indices
        elif self.split_type == 'qke':
            return out_indices, 0    # we assume no delicate because we choose largest margin
        else:
            num_delicate = sum([j[1] for j in out_indices])
            return [j[0] for j in out_indices], num_delicate

    def train(self, data_df, split_type, num_rand_gen, ret_split_list=False, svm_num_train=None, kern_el_dict=None):
        """ This trains the split function, to save PQC with the best splitting as per the criterion. The
            split_type can be either 'qke' or 'random' in line with how we construct the decision tree. The method
            returns the child index for each instance by default. ret_split_list=True, returns instances split. """

        assert num_rand_gen is not None, "Need to supply number of random PQCs/SVMs generated."
        assert split_type != 'qke' or svm_num_train is not None, "Need to supply number of samples to train SVM."
        assert split_type != 'qke' or 'as_params' in self.embed, "Embedding must be 'as_params' for QRF-QKE."
        
        self.svm_num_train = svm_num_train
        self.init_split_type(split_type)
        
        if split_type == 'random':
            random_pqcs = self.pqc.gen_rand(num_rand_gen, param_init_type=self.branch_var)
            self.pqc, childindxs = self.select_best_pqc(pqc_list=random_pqcs, data_df=data_df)
        elif split_type == 'qke':
            childindxs = self.select_best_svm_pqc(data_df, num_train=svm_num_train, num_rand_gen=num_rand_gen,
                                                  pqc_arch_type=self.branch_var, kern_el_dict=kern_el_dict)
        else:
            print("Need valid split_type."); exit()

        out = self.convert_indxs_to_df_list(data_df, childindxs) \
                             if ret_split_list else childindxs

        self._trained_flag = True
        return out
    
    def kernel(self, X1, X2, parallel=False, kern_el_dict=None):
        """ X1 and X2 are matrices where each row is an instance."""

        X = self._gen_X_for_kernel_pqc(X1, X2) # This gets all pairs

        if self._flag_qsvm and self.kern_el_dict_for_qsvm is not None:
            kern_el_dict = self.kern_el_dict_for_qsvm
        
        if parallel:
            pqc_out = _parallel_compute_kernel(X, self.pqc, cores=parallel, kern_el_dict=kern_el_dict)
        else:
            pqc_out = self.pqc(X, ret_prob_0=True, kern_el_dict=kern_el_dict)
        
        K = np.array(pqc_out).reshape((len(X1), len(X2)))

        if self._svm_sym:
            print("We are not looking at subsets of qubits anymore.....");exit()
            X = self._gen_X_for_kernel_pqc(X2, X1) # This gets all pairs
            pqc_out = self.pqc(X, ret_prob_0=True)
            K1 = np.array(pqc_out).reshape((len(X2), len(X1)))
            K = (K + K1.T) / 2
        
        # displ_kernel(K);exit(324)

        if ENFORCE_PSD and self.pqc_sample_num != 'exact' and np.array_equal(X1, X2):
            K = enforce_psd(K)
        if K.shape == (1,1):
            K = np.float(K.squeeze())
        
        # if not self._trained_flag and self._train_kernel is None:
        #     self._train_kernel = K
        # else:
        #     print("K", K)

        return K
    
    # def nystrom_kernel(self, X1, X2): poop
    #     """ This returns the kernel as well storing the kernel elements that need not be recalculated
    #         when splitting down the tree. """
    #     assert not ENFORCE_PSD, "Cannot enforce PSD while using the Nystrom method."
            
    #     if not np.array_equal(X1, X2):
    #         if not self._trained_flag:
    #             # print(self._recent_kernel_calc[:, self._svm.support_])
    #             # return self._recent_kernel_calc
    #             return self._recent_kernel_calc[self._svm.support_, :]
    #         return self.kernel(X1=X1, X2=X2)

    #     subset_X = np.array(X2)[:self.svm_num_train, :]
    #     C = self.kernel(X1, subset_X)
    #     self._recent_kernel_calc = nystrom_kernel_approx(C, subset_num=self.svm_num_train)
    #     return self._recent_kernel_calc

    def target_alignment(self, X, y, kernel=None):
        """ This method returns the target alignment, which is a measure of the effectiveness of
            a specific kernel based on its ability to group instances of the same class. 
            Ref: 'Training Quantum Embedding Kernels on Near-Term Quantum Computers'."""
        assert len(X) == len(y), "Number of labels and instances must match."

        d = len(y)
        y = transform_y_labels_to_pm1(y) # From 0,1 we transform to -1,+1
        print("Transformed y labels are:", y)
        K = self.kernel(X, X) if kernel is None else kernel
        K = np.linalg.inv(K) 
        print("Inverse K:", K)
        s = sum([sum([y[i]*y[j]*K[i,j] for j in range(d)]) for i in range(d)])
        print("s is", s)
        K_norm = np.sqrt((K*K).sum())
        print("K_norm is",K_norm)
        return s/(d*K_norm)

    def svm_predict(self, instances, kern_el_dict=None):
        """ This method applies the split and returns indicies."""
        # This prediction will give two classes -- indicating branch 
        if self.nystrom:
            if not self._trained_flag:
                instances = self.feat_map_nystrom._prev_transform
            else:
                instances = self.feat_map_nystrom.transform(list(instances), kern_el_dict=kern_el_dict)
        return self._svm.predict(instances)  
    
    def select_best_svm_pqc(self, data_df, num_train=None, num_rand_gen=None, pqc_arch_type=None,
                            kern_el_dict=None):
        """ Also returns child indexs for the best split. """
        orig_num = num_rand_gen
        best_svm, max_crit, best_indxs, best_pqc, i = None, None, None, None, 0
        svm_c = self.svm_c
        while i < num_rand_gen:
            if pqc_arch_type == 'eff_anz_pqc_arch':
                self.pqc = self.pqc.ret_eff_anz_qke_pqc()
            elif pqc_arch_type == 'qke_pool':
                self.pqc = self.pqc.ret_qke_pqc_pool()
            elif pqc_arch_type == 'iqp_anz_pqc_arch':
                self.pqc = self.pqc.ret_iqp_anz_qke_pqc()
            elif pqc_arch_type == 'pqc_arch':
                self.pqc = self.pqc.ret_rand_arch_qke_pqc()    # gets a new architecture
            else:
                print("Has not yet been implemented."); exit()
            self.pqc.modify_to_qke_observables(self.num_rand_meas_q)  # gets new qubits measured
            self._svm = self._ret_svm(svm_c=svm_c)

            if self.nystrom:
                self.reset_nystrom_transform()
            self._svm_train(data_df, num_train=num_train, kern_el_dict=kern_el_dict)
            indxs = self(data_df.X, kern_el_dict=kern_el_dict)
            # print(indxs)
            # print(list(data_df.y))
            split_df_list = self.convert_indxs_to_df_list(data_df, indxs)
            crit = self.criterion(data_df, split_df_list)
            print("Info gain: {:.4f}".format(crit))
            print("Accuracy for binary dataset: {:.4f}".format(sum(np.array(indxs)==np.array(data_df.y))/len(indxs)))
            print("Number of SV: {}".format(self._svm.n_support_))

            if max_crit is None or crit > max_crit:
                max_crit = crit
                best_svm = self._svm
                best_indxs = transform_indxs_from_shuffle(data_df, indxs)
                best_pqc = self.pqc

            if max_crit == 0:
                print("Increase SVM_C...")
                svm_c = svm_c * 10
                num_rand_gen += 1
            if i > 2 * orig_num:
                break
            i += 1

        self._svm = best_svm
        self.pqc = best_pqc
        print("----> Selected SVM info gain: {:.4f}".format(max_crit))

        return best_indxs

    def _svm_train(self, data_df, num_train=None, kern_el_dict=None):
        """ This method trains the SVM with a subset of the instances. """
        if self.nystrom:
            X = np.array(list(data_df.X.values)) 
            # Randomly chooses landmark points and constructs the Nystrom embedding.
            X = self.feat_map_nystrom.fit_transform(X, kern_el_dict=kern_el_dict) 
        
            self.select_best_base_class_and_train(X, data_df)
        else:
            X, y = self.select_training_instances(data_df, num_train)
            X = X.values
            self._svm.fit(list(X), y)
    
    def select_training_instances(self, data_df, num_train=None):
        num_train = num_train if num_train is not None else len(data_df)
        assert num_train <= len(data_df), "Not enough training data to train SVM."
        
        base_class, other_class = ret_binary_class(data_df)
        
        comb_df = data_df[(data_df.y == base_class) | (data_df.y == other_class)]
        # comb_df = data_df
        samples = comb_df.sample(n=num_train, replace=False)
        X = samples.X
        y = [0 if yy==base_class else 1 for yy in samples.y]

        if sum(y) == 0: # i.e there is only one class in the sample, that is the base
            other_class_df = comb_df[comb_df.y == other_class]
            X.iloc[-1] = other_class_df.sample().iloc[0].X 
            y[-1] = 1 
        elif sum(y) == num_train: # i.e there is only one class in the sample, that is not the base
            base_class_df = comb_df[comb_df.y == base_class]
            X.iloc[-1] = base_class_df.sample().iloc[0].X
            y[-1] = 0
        
        return X, y
    
    def select_best_base_class_and_train(self, X_trans, data_df):
        classes = np.unique(data_df.y)
        np.random.shuffle(classes)
        best_svm, max_ig = None, 0

        if NYSTROM_CLASSICAL_KERNEL != 'linear' or len(classes) <= 2:
            base_class, _ = ret_binary_class(data_df)
            y = [0 if yy==base_class else 1 for yy in data_df.y]
            self._svm.fit(list(X_trans), y)
            return 

        elif CLASS_SELECTION == 'OAA': # One-against-all
            for base_class in classes[:]:
                y = [0 if yy==base_class else 1 for yy in data_df.y]
                svm = self._ret_svm()
                svm.fit(list(X_trans), y)
                pred = svm.predict(list(X_trans))
                ig = self.criterion(data_df, [data_df.iloc[pred == i] for i in range(2)])
                print(f"IG (class {base_class}): {ig}")
                # margin = 1 / np.sqrt(np.sum(svm.coef_ ** 2))
                # print("Margin (class {}): {}".format(base_class, margin))
                if ig >= max_ig:
                    max_ig = ig
                    best_svm = svm
            self._svm = best_svm
                
        elif CLASS_SELECTION == 'split':
            c1 = classes[:int(len(classes)/2)]
            y = [0 if yy in c1 else 1 for yy in data_df.y]
            self._svm.fit(list(X_trans), y)
            return 

        else:
            print("Error in class selection.");exit()

    def select_best_ob_q(self, data_df, pqc):
        """ This method selects the best qubits to measure in order to maximise the criterion. 
            Once selected it modifies pqc and indxs, num_delicate. """
        if self.num_rand_meas_q is None or self.num_rand_meas_q == 1:
            return self(data_df.X, pqc=pqc, _training=True)
        observed_q_pool = np.random.choice(pqc.n_qubits, size=self.num_rand_meas_q, replace=False)
        observed_q_pool.sort()
        pqc._modify_observed_q(observed_q_pool)
        counts_list = pqc.get_counts(data_df.X)
        best_observed_q_label, indxs, num_delicate = self._select_observe_q(data_df=data_df, pqc=pqc, 
                                                        counts_list=counts_list, 
                                                        observed_q_pool=observed_q_pool)

        pqc._modify_observed_q(best_observed_q_label)
        return indxs, num_delicate

    def convert_indxs_to_df_list(self, data_df, childindxs):
        assert len(data_df) == len(childindxs), "Num data must match indices, {} != {}".format(len(data_df), len(childindxs))
        childindxs = np.array(childindxs)
        return [data_df[childindxs == child_indx] for child_indx in range(self.split_num)]

    def select_best_pqc(self, pqc_list, data_df):
        """ Selects best PQC in list as per the criterion. This is used for 'random' dt_type. """
        max_crit, childindxs, i = None, None, 0
        while i < len(pqc_list):
            indxs, num_delicate = self.select_best_ob_q(data_df, pqc=pqc_list[i]) # This modifies the given pqc to have the best measurement q
            if num_delicate > 0:
                if num_delicate < len(data_df) * DELICACY_PROP:
                    print("Some delicate results from PQC ({:.2f}%). But allowed...".format(100 * num_delicate / len(data_df)))
                else:
                    print("PQC results in a delicate split function ({:.2f}%). Finding new PQC architecture..."\
                                    .format(100 * num_delicate / len(data_df)))
                    pqc_list[i] = self.pqc.ret_rand_arch_pqc()
                    continue
            split_df_list = self.convert_indxs_to_df_list(data_df, indxs)
            crit = self.criterion(data_df, split_df_list)
            
            if max_crit is None or crit > max_crit:
                childindxs = indxs
                max_crit_pqc_indx = i
                max_crit = crit
            i += 1

        print("----> Selected PQC info gain: {:.4f}".format(max_crit))
        return pqc_list[max_crit_pqc_indx], childindxs

    def copy(self, split_num=None, dt_type=None):
        """ This copies the SplitFunction with possibly a different number of splits. """

        split_num = split_num if split_num is not None else self.split_num
        if dt_type == 'qke':
            if self.branch_var == 'eff_anz_pqc_arch':
                num_layers = self.pqc.num_params // self.pqc.n_qubits
                return self.init_eff_anz_pqc_qke(self.pqc.n_qubits, num_layers, self.criterion, split_num, 
                                self.pqc_sample_num, self.embed, self.branch_var, self.num_rand_meas_q, self.nystrom, self.svm_c)
            elif self.branch_var == 'iqp_anz_pqc_arch':
                return self.init_iqp_anz_pqc_qke(self.pqc.n_qubits, self.pqc.num_params, self.criterion, split_num, 
                                self.pqc_sample_num, self.embed, self.branch_var, self.num_rand_meas_q, self.nystrom, self.svm_c)
            elif self.branch_var == 'qke_pool':
                return self.init_pqc_qke_pool(self.pqc.n_qubits, self.pqc.num_params, self.criterion, split_num, 
                                self.pqc_sample_num, self.embed, self.branch_var, self.num_rand_meas_q, self.nystrom, self.svm_c)
            elif self.branch_var == 'pqc_arch':
                return self.init_rand_pqc_qke(self.pqc.n_qubits, self.pqc.num_params, self.criterion, split_num, 
                                self.pqc_sample_num, self.embed, self.branch_var, self.num_rand_meas_q, self.nystrom, self.svm_c)
            else:
                print("Has not yet been implemented."); exit()
        pqc = self.pqc.gen_rand(1, param_init_type=self.branch_var)[0]
        return SplitFunction(self.criterion, split_num, pqc, self.pqc_sample_num, self.embed,
                             branch_var=self.branch_var, num_rand_meas_q=self.num_rand_meas_q, nystrom=self.nystrom, 
                             svm_c=self.svm_c)

    def map_pqc_out(self, pqc_out, _training=False):
        """ This method maps the output of the pqc to an index indicating the child node. """

        assert abs(pqc_out) <= 1, "Error in the output of PQC. Output should be between -1 and 1. "
        mapping = np.arange(self.split_num) * 2 / self.split_num - 1
        for i in range(self.split_num):
            if mapping[i] > pqc_out:
                # print(mapping[i] - pqc_out, 'and', abs(pqc_out - mapping[i-1]), 'for', self.pqc.observed_q)
                if _training:
                    if ((mapping[i] - pqc_out < self._epsilon) or
                                  (abs(pqc_out - mapping[i-1]) < self._epsilon)):
                        return [i - 1, True] # True is identifying that this is delicate
                    else:
                        return [i - 1, False]
                return i - 1

        if not _training:
            return int(self.split_num - 1)
        # The following occurs only during training
        bool_delicate = True if abs(pqc_out - mapping[i]) < self._epsilon else False
        return [int(self.split_num - 1), bool_delicate]

    def _select_observe_q(self, data_df, pqc, counts_list, observed_q_pool):
        ob_dict = {i:q for i, q in enumerate(observed_q_pool)}

        observed_q_options = itertools.combinations(range(len(observed_q_pool)), len(pqc.observed_pauli_str))
        max_crit, max_crit_ob_q, max_crit_indxs, max_crit_num_delicate = 0, None, None, None
        for observed_q in observed_q_options:
            pqc_out_list = exp_val_from_counts_list(counts_list, ob_q=observed_q)
            indxs, num_delicate = self(data_df.X, pqc=pqc, _training=True, _pqc_out_list=pqc_out_list)
            split_df_list = self.convert_indxs_to_df_list(data_df, indxs)
            crit = self.criterion(data_df, split_df_list)
            if crit >= max_crit:
                max_crit = crit
                max_crit_ob_q = observed_q
                max_crit_indxs = indxs
                max_crit_num_delicate = num_delicate

        print("-> Selected qubit info gain: {:.4f}".format(max_crit))
        return [ob_dict[ob_q] for ob_q in max_crit_ob_q], max_crit_indxs, max_crit_num_delicate

    def _gen_X_for_kernel_pqc(self, X1, X2):
        jobs = []
        for x1 in X1:
            for x2 in X2:
                jobs.append(np.concatenate([x1, x2]))
        return jobs 
    
    def _ret_svm(self, verbose=False, max_iter=-1, classical_kernel=False, nystrom=None, 
                svm_c=None, qsvm=False):
        if nystrom is None:
            nystrom = self.nystrom
        if svm_c is None:
            svm_c = self.svm_c

        if classical_kernel:
            return svm.SVC(kernel=classical_kernel, C=svm_c, verbose=verbose, max_iter=max_iter, tol=0.00001)
        elif nystrom:
            return svm.SVC(kernel=NYSTROM_CLASSICAL_KERNEL, C=svm_c, verbose=verbose, max_iter=max_iter, tol=0.00001)
        
        s = svm.SVC(kernel=self.kernel, C=svm_c, verbose=verbose, max_iter=max_iter, tol=0.00001)
        if qsvm:
            s._flag_qsvm = True
        return s

    def reset_nystrom_transform(self):
        self.feat_map_nystrom = Nystrom(kernel=self.kernel, num_sample=self.svm_num_train)

    def init_split_type(self, split_type):
        self.split_type = split_type
        self._svm = self._ret_svm() if split_type == 'qke' else None
    
    def reset_sample_num(self, sample_num):
        self.pqc_sample_num = sample_num
        self.pqc.reset_sample_num(sample_num)
    
    def get_margin(self):
        assert self._trained_flag, "Margins can only be obtained post training of the split function."
        return get_margin(self._svm, True, self.nystrom)
        
    @classmethod
    def init_rand_pqc(cls, n_qubits, num_params, criterion, split_num, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c=None):
        pqc = PQC.init_rand_arch(n_qubits, num_params, pqc_sample_num, embed)
        return cls(criterion, split_num, pqc, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c)

    @classmethod
    def init_eff_anz_pqc(cls, n_qubits, num_layers, criterion, split_num, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c=None):
        pqc = PQC.init_eff_anz(n_qubits, num_layers=num_layers, pqc_sample_num=pqc_sample_num,
                               embed=embed)
        return cls(criterion, split_num, pqc, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c)
    
    @classmethod
    def init_rand_pqc_qke(cls,  n_qubits, num_params, criterion, split_num, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c=None):
        pqc = PQC.init_rand_arch_qke(n_qubits, num_params, pqc_sample_num, embed)
        pqc.modify_to_qke_observables(num_rand_meas_q)
        return cls(criterion, split_num, pqc, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c)

    @classmethod
    def init_eff_anz_pqc_qke(cls, n_qubits, num_layers, criterion, split_num, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c=None):
        pqc = PQC.init_eff_anz_qke(n_qubits, num_layers, pqc_sample_num, embed)
        pqc.modify_to_qke_observables(num_rand_meas_q)
        return cls(criterion, split_num, pqc, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c)
    
    @classmethod
    def init_iqp_anz_pqc_qke(cls, n_qubits, num_params, criterion, split_num, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c=None):
        pqc = PQC.init_iqp_anz_qke(n_qubits, num_params, pqc_sample_num, embed)
        pqc.modify_to_qke_observables(num_rand_meas_q)
        return cls(criterion, split_num, pqc, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c)

    @classmethod
    def init_pqc_qke_pool(cls,  n_qubits, num_params, criterion, split_num, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c=None):
        pqc = PQC.init_qke_pqc_pool(n_qubits, num_params, pqc_sample_num, embed)
        pqc.modify_to_qke_observables(num_rand_meas_q)
        return cls(criterion, split_num, pqc, pqc_sample_num, embed, branch_var, num_rand_meas_q, nystrom, svm_c)
    

class SplitCriterion:
    """ This object contains a classical criterion used for maximising the effectiveness of a particular split.
        The idea is that this contains various criterions for the both continuous and discrete cases. """
    def __init__(self, split_crit_name, model_type):
        self.split_crit_name = split_crit_name
        self.model_type = model_type                    # Either 'clas' or 'reg'
        self.fn = None                                  # This is the function that returns the criterion

    def __call__(self, initial_df, split_df_list):
        """ This returns the goodness of the split when given a list of dataframes. """
        return self.fn(initial_df, split_df_list)

    @classmethod
    def init_gini(cls):
        pass

    @classmethod
    def init_info_gain(cls, model_type):
        sp = cls('info_gain', model_type)
        if model_type == 'clas':
            crit_fn = cls._clas_crit_fn
        elif model_type == 'reg':
            crit_fn = cls._reg_crit_fn
        else:
            crit_fn = None
            print("Has not yet been implemented."); exit()

        sp.fn = crit_fn
        return sp

    @classmethod
    def init_entropy(cls):
        pass
    
    @staticmethod
    def _clas_crit_fn(initial_df, split_list):
        init_class_probs = return_class_probs(initial_df)
        children_class_probs = [return_class_probs(branch) for branch in split_list]
        init_entropy = calc_entropy(init_class_probs)
        child_entropies = np.array([calc_entropy(child_class_probs)
                                    for child_class_probs in children_class_probs])
        weighting = np.array([len(s) for s in split_list]) / len(initial_df)
        return init_entropy - np.dot(child_entropies, weighting)
    
    @staticmethod
    def _reg_crit_fn(initial_df, split_list):
        init_reg_var = return_label_variance(initial_df)
        children_reg_var_list = np.array([return_label_variance(branch) for branch in split_list])
        return init_reg_var - children_reg_var_list.sum()


class ImpossibleSVMTrainError(Exception):
    pass

# ----------------------------------------------------------------------------------
# -------------------------- Useful functions --------------------------------------


def get_margin(svm, include_margin, nystrom):
    # assert not include_margin or (nystrom and include_margin), "Margin can only be returned with Nystrom. "
    if include_margin and nystrom:
        margin = 1 / np.sqrt(np.sum(svm.coef_ ** 2))
        return margin
    else:
        return -1


def calc_entropy(class_probs):
    class_probs = np.array(class_probs)
    return - np.dot(class_probs, np.log2(class_probs))


def return_class_probs(class_probs_df):
    """ Given a Dataframe we return the probabilities of each class in data set. """
    classes = class_probs_df.y
    tot_num = len(classes)
    unique_classes = classes.unique()
    return [(class_probs_df.y == c).sum() / tot_num for c in unique_classes]


def transform_y_labels_to_pm1(y):
    """ From 0,1 we transform to -1,+1. """
    return (np.array(y)*2) - 1 


def return_label_variance(df):
    """ Given a dataframe, this function returns the variance of the continuous labels. """
    labels = df.y
    return np.var(np.array(labels))


def compute_classical_kernel(kernel_type, X):

    if kernel_type == 'rbf':
        kernel = 1.0 * RBF(1.0)
        return kernel(list(X))
    else:
        print("Has not yet been implemented."); exit()


def _parallel_compute_kernel(X, pqc, cores, kern_el_dict=None):
    num_cpu = cores if isinstance(cores, int) else os.cpu_count()
    print(f"Using {num_cpu} cores.")
    chunked_X = list(mit.chunked(X, np.ceil(len(X)/num_cpu).astype(int)))
    arguments = [[ensemb_id, kern_el_dict, pqc, chunked_X[ensemb_id]] for ensemb_id in range(num_cpu)]
    pqc.device = None
    with Pool(num_cpu) as p:
        out = p.map(_parallel_compute_kernel_, arguments)
    print(out)
    out.sort()
    pqc.reset_device()
    return [o[1] for o in out] # It will be reshaped anyway
    

def _parallel_compute_kernel_(arg):
    ensemb_id, kern_el_dict, pqc, x = arg
    pqc.reset_device()
    print(f"ensemb_id={ensemb_id}")
    out = pqc(x, ret_prob_0=True, kern_el_dict=kern_el_dict)
    return [ensemb_id, out]


def relabel_dataset(dataset, sf, data_df, save=True, parallel=False, qrf_optimal=False, force_relabel=False):
    """ This function relabels the dataset for separation and then saves it. This modifies data_df."""
    fname = relabelled_filename(dataset, sf.pqc.num_params, sf.embed, qrf_optimal=qrf_optimal)
    print("Relabelling into {} .... ".format(fname))
    if parallel:
        print(f"Running in parallel with {parallel} cores")
    
    # we check if the kernel has already been done 
    if check_dataset_relabelled(dataset, sf.pqc.num_params, sf.embed, qrf_optimal=qrf_optimal, kernel=True) and not force_relabel:
        print("Loaded pre-existent kernel and associated dataset.")
        f = unpickle_file(relabelled_filename(dataset, sf.pqc.num_params, sf.embed, qrf_optimal=qrf_optimal, kernel=True))
        print("WARNING: Dataset has been re-uploaded.")
        q_kernel = f['kernel']
        print("WARNING: Data being obtained from the kernel file.")
        data_df = f['data_df'] 
    else:
        print("Computing quantum kernel...")
        q_kernel = sf.kernel(data_df.X, data_df.X, parallel=parallel)
    c_kernel = compute_classical_kernel('rbf', data_df.X)
    new_y = get_stilted_dataset(q_kernel, c_kernel, qrf_optimal=qrf_optimal)
    data_df['orig_y'] = data_df['y']
    data_df['y'] = new_y
    if save:
        save_dataset_relabelled(data_df, dataset, sf.pqc.num_params, sf.embed, qrf_optimal=qrf_optimal)
        save_kernel_for_dataset_relabelled(data_df, q_kernel, dataset, sf.pqc.num_params, sf.embed, qrf_optimal=qrf_optimal)
    return data_df


def convert_to_qrf_opt(dataset, n_qubits, embed):
    """ Ensure that if you are going to use this function that the kernel has been stored. """
    print("\nRelabelling QRF-optim for (dataset, n_qubits, embed)=({}, {}, {}).".format(dataset, n_qubits, embed))
    criterion = SplitCriterion.init_info_gain('clas')
    if embed == 'iqp' or embed == 'as_params_iqp':
        num_params = n_qubits*(n_qubits+1)
        sf = SplitFunction.init_iqp_anz_pqc_qke(n_qubits, num_params=num_params, criterion=criterion, 
                                        split_num=2, pqc_sample_num='exact', embed='as_params_iqp', 
                                        branch_var='iqp_anz_pqc_arch', num_rand_meas_q=n_qubits, nystrom=True,
                                        svm_c=5)
        embed_type = 'as_params_iqp'
    elif embed == 'eff' or embed == 'as_params_all':
        num_params = 2 * n_qubits**2
        num_layers = np.ceil(num_params / n_qubits).astype(int)
        sf = SplitFunction.init_eff_anz_pqc_qke(n_qubits, num_layers=num_layers, criterion=criterion, 
                                        split_num=2, pqc_sample_num='exact', embed='as_params_all', 
                                        branch_var='eff_anz_pqc_arch', num_rand_meas_q=n_qubits, nystrom=True,
                                        svm_c=5)
        embed_type = 'as_params_all'
    
    data_df = all_datasets(dataset=dataset, ret_embed=False, train_prop=0.9999,
                                         load_relabelled=False, embed=embed_type, num_params=num_params, X_dim=n_qubits)[0] 
    relabel_dataset(dataset, sf, data_df, save=True, parallel=False, 
                    qrf_optimal=True)


def change_delicacy(del_prop):
    global DELICACY_PROP
    DELICACY_PROP = del_prop


def get_delicacy():
    return DELICACY_PROP


def ret_binary_class(data_df):
    try:
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        base_class = int(data_df.iloc[np.random.randint(len(data_df))].y)
        other_data = data_df[data_df.y != base_class]
        other_class = int(other_data.iloc[np.random.randint(len(other_data))].y)
    except ValueError:
        raise ImpossibleSVMTrainError

    return base_class, other_class


def shuffle_data_df(data_df):
    """ This function shuffles the data for Nystrom that takes the first few instances. """
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    return data_df.sample(frac=1).reset_index(drop=False)


def transform_indxs_from_shuffle(data_df, indxs):
    if 'index' not in data_df:
        return indxs        # There was no transformation
    else:
        assert 'level_0' not in data_df, "Error in unshuffling."
        print('before', indxs)
        i = list(data_df.sort_values('index').reset_index(drop=False)['level_0'])
        print('after', indxs[i])
        return indxs[i]


def displ_kernel(K):
    im = plt.imshow(K, norm=colors.LogNorm())
    plt.colorbar(im)
    plt.show()
    print('Kernel:\n',K)
    try:
        e = np.linalg.eigvals(K)
        print("Eigenvalues of the kernel: {}".format(e))
        print("With sum={}".format(sum(e)))
    except:
        pass


def enforce_psd(K):
    D, U = np.linalg.eig(K)  
    K = U @ np.diag(np.maximum(0, D)) @ U.transpose()

    return K 


def nystrom_kernel_approx(C, subset_num): # Not in use
    W = C[:subset_num, :]
    D, U = np.linalg.eig(W)  
    print('eigenvalues:',D)
    W_plus = sum([np.real(D[i]**-1) * np.dot(U[i:i+1].T, U[i:i+1]) for i in range(len(D))])

    K_approx = C @ W_plus @ C.T
    return K_approx