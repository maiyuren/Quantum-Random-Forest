
import numpy as np
from pickle_functions import unpickle_file
from data_construction import relabelled_filename, check_dataset_relabelled
from pqc import get_embed_transformed_states
import tqdm 

class KernelElementStorage:
    def __init__(self) -> None:
        self.counts_storage = dict()

    def check(self, embed, X):
        embed_storage = self.ret_storage_for_embed(embed)
        return self.check_in_storage(storage=embed_storage, X=X) or self.check_in_storage(storage=embed_storage, X=X, reverse=True)

    def ret_counts(self, embed, X):
        """ Make sure you checked first, otherwise you will throw an error. """
        embed_storage = self.ret_storage_for_embed(embed)
        if self.check_in_storage(storage=embed_storage, X=X):
            return embed_storage[tuple(X)]
        else:
            return embed_storage[tuple(self._reverse_vec(X))]

    def add_counts(self, embed, X, counts):
        if self.check(embed, X):
            return
        if embed not in self.counts_storage:
            self.counts_storage[embed] = {}
        self.counts_storage[embed][tuple(X)] = counts

    def check_in_storage(self, storage, X, reverse=False):
        assert not len(X) % 2, "Two states must have been concatenated at this point. Error."
        if reverse:
            X = self._reverse_vec(X)
        return tuple(X) in storage
    
    def ret_storage_for_embed(self, embed):
        if embed not in self.counts_storage:
            self.counts_storage[embed] = {}
        return self.counts_storage[embed]

    def size(self):
        return {k: len(self.counts_storage[k]) for k in self.counts_storage.keys()}

    def delete_storage(self):
        keys = tuple(self.counts_storage.keys())
        for key in keys:
            del self.counts_storage[key]

    @staticmethod
    def _reverse_vec(X):
        return np.concatenate([X[len(X)//2:], X[:len(X)//2]])

    @classmethod
    def generate_from_kernel_matrix(cls, data_df, kernel_mat, embed, num_params, n_qubits, pqc_sample_num):
        kern_el_dict = cls()
        X = np.array(list(data_df.X))
        for i1, x1 in tqdm.tqdm(enumerate(X), total=len(X)):
            for i2, x2 in enumerate(X):
                x = np.concatenate([x1, x2])
                k = kernel_mat[i1, i2]
                if i1 != i2:
                    # Add gaussian noise 
                    k = k + np.random.normal(scale=np.sqrt(k*(1-k)/pqc_sample_num))
                    if k < 0:
                        k = 0
                    elif k > 1:
                        k = 1
                num_zero = np.round(k * pqc_sample_num).astype(int)
                num_non_zero = pqc_sample_num - num_zero
                counts = {'0' * n_qubits: num_zero, 
                          '1' * n_qubits: num_non_zero}
                
                x = get_embed_transformed_states([x], embed=embed, n_qubits=n_qubits, 
                                                num_params=num_params)[0]
                kern_el_dict.add_counts(embed, x, counts=counts)

        return kern_el_dict
    
    @classmethod
    def generate_from_kernel_matrix_file(cls, dataset, embed, num_params, n_qubits, pqc_sample_num):
         
        kernel_filename = relabelled_filename(dataset, num_params, embed, kernel=True)
        print("Loading kernel elements from file:", kernel_filename)
        if check_dataset_relabelled(dataset, num_params, embed, kernel=True):
            f = unpickle_file(relabelled_filename(dataset, num_params, embed, kernel=True))
            kernel_mat = f['kernel']
            data_df = f['data_df']
        else: 
            print("ERROR: Cannot find kernel file.");exit(342)
            
        assert len(kernel_mat) == len(kernel_mat[0]), "Kernel matrix must be a square matrix."

        return cls.generate_from_kernel_matrix(data_df, kernel_mat, embed, num_params, n_qubits, pqc_sample_num)