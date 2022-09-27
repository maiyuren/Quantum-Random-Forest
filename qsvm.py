
import numpy as np
from kernel_element_storage import KernelElementStorage
from pickle_functions import unpickle_file

class QSVM:
    """ This QSVM object is tailored towards its comparison with QRF. Therefore it contains
        many of the modules used by the QRF. """
    def __init__(self, qsvm, split_fn=None) -> None:
        
        self.kernel_matrix = None 
        self.qsvm = qsvm
        self.split_fn = split_fn # If given, this is the split function object used inside the qsvm
        self.kern_el_dict = None

    def train(self, data_df):
        X, y = data_df.X, data_df.y
        assert len(X) == len(y), "The number of instances must be the same as the number of labels. "

        self.qsvm.fit(list(X.values), y)

    def predict(self, instances):
        return self.qsvm.predict(instances) 

    def load_kernel_el_to_storage_from_file(self, dataset):
        assert self.split_fn is not None, "Split function must have been loaded. "
        
        pqc_sample_num = self.split_fn.pqc_sample_num
        embed = self.split_fn.embed 
        n_qubits = self.split_fn.pqc.n_qubits     
        num_params = self.split_fn.pqc.num_params   

        self.split_fn.kern_el_dict_for_qsvm = KernelElementStorage.generate_from_kernel_matrix_file(dataset, 
                                                        embed=embed, num_params=num_params,
                                                        n_qubits=n_qubits, pqc_sample_num=pqc_sample_num)
        self.kern_el_dict = self.split_fn.kern_el_dict_for_qsvm