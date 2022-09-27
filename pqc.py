
from pickle_functions import pickle_file, unpickle_file
from itertools import count
import tqdm
from pandas.core.indexes.api import union_indexes
import numpy as np
import cirq
import sympy
import copy
import random
import pandas as pd
from device import get_device
import os

_PARAM_COUNT = 0
PAULI_STR = ('Z',)# 'ZZ')#, 'ZZZ')    # Making ZZ more likely
DEVICE_ARCHITECTURE = 'line'
SEED = None                     # This sets the seed for reproducing results. None, sets to random

KERNEL_POOL_FILENAME = "kernel_pqc_pool/pool_of_kernels.pickle"
KERNEL_POOL_FILE = None

import matplotlib.pyplot as plt

class PQC:
    """ This is the Parameterised quantum circuit object where all quantum computation occurs. """
    def __init__(self, n_qubits, num_params, pqc_sample_num, embed=False, embed_circ=None, embed_params=None):
        self.n_qubits = n_qubits
        self.qubits = self.init_qubits()
        self.num_params = num_params
        self.pqc_sample_num = pqc_sample_num
        self.embed = embed
        self._qke_flag = False
        self._qke_rand = False    # This is a flag used to determine if you finish with a Hadamard

        self.params = [get_param() for _ in range(num_params)]
        self.param_vals = ret_rand_param_vals(num=num_params)
        self.observed_pauli_str = None    # You dont actually use the fact that there could be different Pauli operators
        self.observed_q_label = None
        self.observed_q = None
        self.applied_operators = []
        self.orig_param_list = None
        self.embed_param_indxs_orig = None

        self.init_observables()

        self.circ = cirq.Circuit()
        self.device = reset_device(sample_num=self.pqc_sample_num)

        # Embedding
        self.embed_circ = embed_circ if embed is not False and 'as_params' not in embed else None
        self.embed_params = embed_params if embed is not False and 'as_params' not in embed else None
        self.mapping_matrix = None # Only used for 'as_params_all' embedding

    def __call__(self, states, ret_prob_0=False, kern_el_dict=None):
        """ This method runs the PQC and returns a real number for each state in the list of states. If ret_prob_0=True,
            then the probability of the all 0 measurement is returned - for QKE. """
        counts_list = self.get_counts(states, kern_el_dict=kern_el_dict)
        if ret_prob_0:
            return prob_0_from_counts_list(counts_list, self.pqc_sample_num)
        return exp_val_from_counts_list(counts_list)

    def get_counts(self, states, kern_el_dict=None):
        """ This returns the counts for each of the inputs given when state is sampled."""
        if isinstance(states, pd.Series):
            states.reset_index()

        if self.device is None: # This is if you have just unpickled the file
            self.device = reset_device(sample_num=self.pqc_sample_num)

        example_state = states.iloc[0] if isinstance(states, pd.Series) else states[0]
        if self.embed is not False and 'as_params' in self.embed:
            if self.embed_circ is None: # This will happen at training only
                self.init_embed_circ(example_input=example_state) # This also changes the original self.params
            
            states = get_embed_transformed_states(states, self.embed, self.n_qubits, self.num_params, qke_flag=self._qke_flag)
            # if self.embed == 'as_params_all':
            #     states = self.repeat_params_map(states, len_example_state=len(example_state), num_params=self.num_params, qke_flag=self._qke_flag)
            # elif self.embed == 'as_params_iqp':
            #     assert self.n_qubits == len(example_state) // 2, "Number of qubits must equal the dimesion: {}!={}".format(self.n_qubits, len(example_state)//2)
            #     states = self.iqp_transform_data(states, self.n_qubits)

        elif self.embed:
            if self.embed_circ is None: # This will happen at training only
                self.init_embed_circ(example_input=example_state)
            base_circ = self.embed_circ + self.param_resolver()
        else:
            run_circ = self.param_resolver()

        counts_list = []
        for state in tqdm.tqdm(states):
            if kern_el_dict is not None and kern_el_dict.check(self.embed, state):
                counts_list.append(kern_el_dict.ret_counts(self.embed, state))
                continue

            # The following conditions depend on the type of encoding
            if self.embed is not False and 'as_params' in self.embed:
                run_circ = self.param_resolver(embed_as_params_vals=state)
                results = self.device.run(run_circ, observed_q_label=self.observed_q_label, qubits=self.qubits)
            elif self.embed:
                run_circ = self.param_resolver(self.embed_params, state, base_circ)
                results = self.device.run(run_circ, observed_q_label=self.observed_q_label, qubits=self.qubits)
            else: # Goes into here if the input is a statevector and real device cannot be used
                assert len(state) == 2 ** self.n_qubits, "The given state must be of correct dimension."
                results = self.device.run(run_circ, initial_state=state)

            counts_list.append(self.device.get_counts_from_results(results=results, observed_q_label=self.observed_q_label,
                                                         seed=return_seed()))
            if kern_el_dict is not None and self.device.dev_type == 'cirq':
                kern_el_dict.add_counts(self.embed, state, counts_list[-1])

        # Now if you are running on IBM, the device object would have stored the circuits without executing.
        #   Now they need to be executed
        if self.device.dev_type == 'ibm' and self.device.bulk_jobs:
            counts_list = self.device._run_stored_circ()

        return counts_list

    def param_resolver(self, params=None, param_vals=None, circ=None, embed_as_params_vals=None):
        """ This method returns the circuit with the parameters resolved using self.param_vals. """
        params = self.params if params is None else params
        param_vals = self.param_vals if param_vals is None else param_vals
        circ = self.circ if circ is None else circ
        assert len(params) == len(param_vals), "The length of parameters and their values must match. " \
                                               "({} != {})".format(len(params), len(param_vals))
        c_param_dict = {params[i]:param_vals[i] for i in range(len(params))}
        if embed_as_params_vals is not None:
            assert len(embed_as_params_vals) == len(self.embed_params), "Embedding must match available circ parameters. {} != {}".format(len(embed_as_params_vals), len(self.embed_params))
            em_param_dict = {self.embed_params[i]:embed_as_params_vals[i] for i in range(len(embed_as_params_vals))}
            resolver = cirq.ParamResolver(merge_dict(c_param_dict, em_param_dict))
        else:
            resolver = cirq.ParamResolver(c_param_dict)
        return cirq.resolve_parameters(circ, resolver)

    def gen_rand(self, num_gen, param_init_type='param_rand'):
        """ This generates a number of PQCs who have the same architecture but different parameters and observables. """
        if param_init_type == 'param_rand':
            param_vals_list = [None for _ in range(num_gen)]
            return [self._copy_with_same_arch(param_vals_list[i]) for i in range(num_gen)]
        elif param_init_type == 'pqc_arch':
            return [self.ret_rand_arch_pqc() for i in range(num_gen)]
        elif param_init_type == 'haar':
            print("This has not yet been implemented."); exit()
        else:
            print("Was given:", param_init_type)
            print("Error in parameter value initialisation."); exit()

    def _copy_with_same_arch(self, param_vals=None):
        """ This method copies the PQC object keeping the same architecture - changing meas. observ. and param vals."""
        new_pqc = copy.copy(self)
        new_pqc.param_vals = param_vals if param_vals is not None else ret_rand_param_vals(len(self.params))
        new_pqc.init_observables()          # Randomises the observables
        return new_pqc

    def init_qubits(self):
        if DEVICE_ARCHITECTURE == 'grid':
            print("Has not yet been implemented."); exit()
        if DEVICE_ARCHITECTURE == 'line':
            return cirq.LineQubit.range(self.n_qubits)

    def init_observables(self):
        self.observed_pauli_str = random.choice(PAULI_STR)
        self.observed_q_label = np.random.choice(range(self.n_qubits), len(self.observed_pauli_str), replace=False)
        self.observed_q = [self.qubits[i] for i in self.observed_q_label]
    
    def modify_to_qke_observables(self, num_q=None):
        num_q = self.n_qubits if num_q is None else num_q
        observed_q = np.random.choice(self.n_qubits, size=num_q, replace=False)
        observed_q.sort()
        self._modify_observed_q(observed_q)

    def _modify_observed_q(self, observed_q_labels):
        self.observed_q_label = observed_q_labels
        self.observed_q = [self.qubits[i] for i in self.observed_q_label]

    def init_embed_circ(self, example_input=None):
        num_embed_params = self._req_param_num_embed(example_input)
        num_embed_params = num_embed_params // 2 if self._qke_flag else num_embed_params  # We will double the circuit later in this method
        if self.embed == 'as_params':
            self._replace_for_as_params_embed(num_embed_params) 
        elif self.embed == 'as_params_all':
            self._replace_for_as_params_all_embed()
        elif self.embed == 'as_params_iqp':
            self._replace_for_as_params_iqp_embed()
        else:
            self.embed_params = [get_param() for _ in range(num_embed_params)]

        if self._qke_flag: # This transforms the circuit to have the conjugate added - hence doubling number of parameters
            self.qke_transform()

        self.embed_circ = self.create_embed_circ(self.embed, self.embed_params)

    def ret_rand_arch_pqc(self):
        return PQC.init_rand_arch(self.n_qubits, self.num_params, self.pqc_sample_num, self.embed,
                                  embed_circ=self.embed_circ, embed_params=self.embed_params)
    
    def ret_rand_arch_qke_pqc(self):
        return PQC.init_rand_arch_qke(self.n_qubits, self.num_params, self.pqc_sample_num, 
                embed=self.embed, embed_circ=self.embed_circ, embed_params=self.embed_params)
    
    def ret_eff_anz_qke_pqc(self):
        num_params = self.num_params // self.n_qubits
        return PQC.init_eff_anz_qke(self.n_qubits, num_params, self.pqc_sample_num, 
                embed=self.embed, embed_circ=self.embed_circ, embed_params=self.embed_params)
    
    def ret_iqp_anz_qke_pqc(self):
        return PQC.init_iqp_anz_qke(self.n_qubits, self.num_params, self.pqc_sample_num, 
                embed=self.embed, embed_circ=self.embed_circ, embed_params=self.embed_params)

    def ret_qke_pqc_pool(self):
        """ From the pool of kernels, this method selects a random kernel from the pool with the correct parameters."""
        return PQC.init_qke_pqc_pool(self.n_qubits, self.num_params, self.pqc_sample_num, 
                embed=self.embed, embed_circ=self.embed_circ, embed_params=self.embed_params)
    
    @staticmethod
    def _ret_qke_pqc_from_pool(n_qubits, num_params, pqc_sample_num, embed):
        d = get_pool_data()  
        assert isinstance(d, pd.DataFrame), "Pool should be a DataFrame object."
        
        d = d[d.n_qubits == n_qubits]
        d = d[d.num_params == num_params]
        d = d[d.embed == embed]
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        i = np.random.randint(len(d))
        pqc = d.iloc[i].pqc
        print("(Loading PQC from pool...{})".format(i))
        pqc.reset_sample_num(pqc_sample_num)
        return pqc
    
    @classmethod
    def init_qke_pqc_pool(cls, n_qubits, num_params, pqc_sample_num, embed=False, embed_circ=None,
                       embed_params=None):
        # assert embed_circ is None and embed_params is None, "QKE does not allow for separate embedding. Given, embed_circ={} and embed_params={}".format(embed_circ, embed_params)
        try:
            return cls._ret_qke_pqc_from_pool(n_qubits, num_params, pqc_sample_num, embed)
        except ValueError:
            print("No appropriate kernel found in pool. Creating random 'rand_arch' QKE PQC.")
            return cls.init_rand_arch_qke(n_qubits, num_params, pqc_sample_num, embed=embed,
                                          embed_circ=embed_circ, embed_params=embed_params) 

    @classmethod
    def init_rand_arch(cls, n_qubits, num_params, pqc_sample_num, embed=False, embed_circ=None,
                       embed_params=None):
        """ This initiates a PQC with a random architecture. """
        pqc = cls(n_qubits, num_params=num_params, pqc_sample_num=pqc_sample_num, embed=embed,
                  embed_circ=embed_circ, embed_params=embed_params)
        pqc._create_random_parameterised_circ()
        return pqc

    @classmethod
    def init_eff_anz(cls, n_qubits, num_layers, pqc_sample_num, embed=False, embed_circ=None,
                       embed_params=None):
        """ This initiates a PQC with the efficient Anzatz architecture. """
        num_params = num_layers * n_qubits
        pqc = cls(n_qubits, num_params=num_params, pqc_sample_num=pqc_sample_num, embed=embed,
                  embed_circ=embed_circ, embed_params=embed_params)
        pqc._create_eff_anz_circ(num_layers)
        return pqc
    
    @classmethod
    def init_rand_arch_qke(cls, n_qubits, num_params, pqc_sample_num, embed=False, embed_circ=None,
                            embed_params=None):
        """ This initiates with random architecture but with QKE in mind. In this case, num_params refers to 
            the number of parameters for each unitary. Hence the total number of parameters=2*num_params. """
        pqc = cls(n_qubits, num_params=num_params, pqc_sample_num=pqc_sample_num, embed=embed,
                  embed_circ=embed_circ, embed_params=embed_params)
        pqc._create_random_parameterised_circ()
        pqc.flag_as_qke()
        pqc._qke_rand = True
        return pqc 

    @classmethod
    def init_eff_anz_qke(cls, n_qubits, num_layers, pqc_sample_num, embed=False, embed_circ=None,
                       embed_params=None):
        num_params = num_layers * n_qubits
        pqc = cls(n_qubits, num_params=num_params, pqc_sample_num=pqc_sample_num, embed=embed,
                  embed_circ=embed_circ, embed_params=embed_params)
        pqc._create_eff_anz_circ(num_layers)
        pqc.flag_as_qke()
        return pqc 

    @classmethod
    def init_iqp_anz_qke(cls, n_qubits, num_params, pqc_sample_num, embed=False, embed_circ=None,
                            embed_params=None):
        pqc = cls(n_qubits, num_params=num_params, pqc_sample_num=pqc_sample_num, embed=embed,
                  embed_circ=embed_circ, embed_params=embed_params)
        pqc._create_iqp_anz_circ(num_params)
        pqc.flag_as_qke()
        return pqc 

    def flag_as_qke(self):
        self._qke_flag = True

    def _create_eff_anz_circ(self, num_layers):
        for i in range(num_layers):
            self._rotation_layer(params=self.params[i * self.n_qubits: (i + 1) * self.n_qubits], layer_num=i)
            self._entan_layer()
    
    def _create_iqp_anz_circ(self, num_params):
        applied_gates = []
        assert num_params == self.n_qubits + self.n_qubits**2, f"It should be: num_params=n + n*n, where n is the number of qubits. We got n_qubits={self.n_qubits}, num_params={num_params}."
        applied_gates.append(["H_all", None, None])
        for i in range(self.n_qubits):
            applied_gates.append(['rot', 'Z', i])
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if i == j:
                    applied_gates.append(['glob_phase', None, i])
                else:
                    applied_gates.append(['ZZ', i, j])
        self._apply_ops_to_circ(applied_gates, init_hadarmard=False)
        self._apply_ops_to_circ(applied_gates, init_hadarmard=False)
        # self._apply_ops_to_circ(applied_gates, init_hadarmard=False)

        self.applied_operators = applied_gates 

    def _rotation_layer(self, params, layer_num):
        # paulis = ('X', 'Y', 'Z')
        for i in range(self.n_qubits):
            # pauli = random.choice(paulis) if layer_num !=0 else 'Y'
            pauli = 'Y' if layer_num % 2 == 0 else 'Z'
            rot = return_pauli_rot(pauli=pauli,
                                   param=params[i])
            self.circ.append(rot(self.qubits[i]))
            self.applied_operators.append(['rot', pauli, i])

    def _entan_layer(self, circ=None):
        """ The CNOTs here will be nearest neighbour and depending on DEVICE_ARCHITECTURE this
            restricts the allowed CNOTs. """
        circ = self.circ if circ is None else circ
        if DEVICE_ARCHITECTURE == 'line':
            init = 0
            for i in range(init, self.n_qubits - 1 + init, 2):  # Loop over even indices: i=0,2,...N-2
                circ.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
                self.applied_operators.append(['cnot', i, i+1])
            for i in range(init + 1, self.n_qubits - 1 + init, 2):  # Loop over odd indices:  i=1,3,...N-3
                circ.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
                self.applied_operators.append(['cnot', i, i+1])
        else:
            print("This has not been implemented.")
            exit()

    def _create_random_parameterised_circ(self):
        """ This creates unitary with random rotations and CNOTs where they equal in number. . """
        paulis = ('X', 'Y', 'Z')
        _qubits = list(range(self.n_qubits))
        # creating CNOT pool randomly
        cnot_pool = []
        while self.n_qubits > 1 and len(cnot_pool) < self.num_params // 1.5:
            contr = np.random.randint(self.n_qubits)
            targ = contr + 2 * np.random.randint(2) - 1
            if targ < 0:
                targ = 1
            if targ >= self.n_qubits:
                targ = self.n_qubits - 2
            cnot_pool.append(['cnot', contr, targ])

        # creating rotation pool -- here we have a weighting to try cover all qubits with rotations
        rot_pool = []
        qubit_count = [1] * self.n_qubits
        while len(rot_pool) < self.num_params:
            q = np.random.choice(_qubits, p=weighting_from_count(qubit_count))
            pauli = random.choice(paulis)
            rot_pool.append(['rot', pauli, q])
            qubit_count[q] += 1

        tot_pool = cnot_pool + rot_pool
        random.shuffle(tot_pool)

        if self.embed == 'simple' or self.embed == 'simple_layered':
            self._apply_ops_to_circ(tot_pool, init_hadarmard=False)
        else:
            self._apply_ops_to_circ(tot_pool, init_hadarmard=True)
        
        self.applied_operators = tot_pool

    def _apply_ops_to_circ(self, pool, init_hadarmard=False, fin_hadarmard=False, params=None, hermitian=False):
        if init_hadarmard:
            self.circ.append(cirq.H(q) for q in self.qubits)
        p_indx = 0
        params = self.params if params is None else params
        herm = -1 if hermitian else 1
        for op in pool:
            if op[0] == 'rot':
                rot = return_pauli_rot(pauli=op[1], param=herm*params[p_indx])
                self.circ.append(rot(self.qubits[op[2]]))
                p_indx += 1
            elif op[0] == 'cnot':
                self.circ.append(cirq.CNOT(self.qubits[op[1]], self.qubits[op[2]]))
            elif op[0] == 'ZZ':
                self.circ.append(cirq.CNOT(self.qubits[op[1]], self.qubits[op[2]]))
                rot = return_pauli_rot(pauli='Z', param=herm*params[p_indx])
                self.circ.append(rot(self.qubits[op[2]]))
                self.circ.append(cirq.CNOT(self.qubits[op[1]], self.qubits[op[2]]))
                p_indx += 1            
            elif op[0] == 'glob_phase':
                self.circ.append(cirq.ops.XPowGate(exponent=0, global_shift=herm*params[p_indx])(self.qubits[op[2]]))
                p_indx += 1   
            elif op[0] == 'H_all':
                self.circ.append(cirq.H(q) for q in self.qubits) 
            else:
                print("Operator not yet implemented.")
                exit()
        
        if fin_hadarmard:
            self.circ.append(cirq.H(q) for q in self.qubits)

        assert p_indx == self.num_params, "Error in parameterising circuit. {}!={}.".format(p_indx, self.num_params)
    
    def _add_meas_ops(self, run_circ):
        assert False not in ["z" == o.lower() for o in self.observed_pauli_str], "Only Z measurements have been implemented."
        run_circ.append(cirq.measure(*self.observed_q))

    def create_embed_circ(self, embed_type, params):
        circ = cirq.Circuit()
        if embed_type == 'simple':
            """ This is when each feature is allocated to a qubit with the rotation identifying 
                value of the feature. Hence the feature is normalised to lie between 0 and 2*pi. """
            assert self.n_qubits == len(params), "This embedding requires: n_qubits == len(vec)."
            for i, q in enumerate(self.qubits):
                rot = cirq.ops.Ry(rads=params[i])
                circ.append(rot(q))
        elif embed_type == 'simple_layered':
            """ This is when we have layers of Ry-CNOT-Rz-CNOT until all parameters have been encoderd. """
            _rot, _layer = [cirq.ops.Ry, cirq.ops.Rz], 0
            for i, p in enumerate(params):
                rot = _rot[_layer % 2](rads=p)
                circ.append(rot(self.qubits[i % self.n_qubits]))
                if i != len(params) - 1 and (i % self.n_qubits) == self.n_qubits - 1:
                    self._entan_layer(circ=circ)
                    _layer += 1
        elif 'as_params' in embed_type:
            return self.circ
        else:
            print("Error in applying {} embedding".format(embed_type)); exit()

        return circ

    def _req_param_num_embed(self, example_input=None):
        if self.embed == 'simple':
            return self.n_qubits
        elif self.embed == 'simple_layered':
            assert example_input is not None, "This embedding requires an example of the input vector."
            return len(example_input)
        elif 'as_params' in self.embed:
            assert example_input is not None, "This embedding requires an example of the input vector."
            assert len(example_input) <= 2 * self.num_params or not self._qke_flag, "This embedding requires more parameters in the split function. {} !< 2*{}".format(len(example_input), self.num_params)
            assert len(example_input) <= self.num_params or self._qke_flag, "This embedding requires more parameters in the split function. {} !< {}".format(len(example_input), self.num_params)
            # assert not self._qke_flag or len(example_input) % 2, "Odd dimensional input. QKE requires concatenated vectors of equal length."
            return len(example_input)
        else:
            print(self.embed)
            print("Has not yet been implemented."); exit()

    def _replace_for_as_params_embed(self, num_embed_params, replace_type='random'):
        # This method is called before the circuit is doubled
        self.orig_param_list = tuple(self.params)
        if replace_type == 'random':
            indxs = np.random.choice(self.num_params, size=num_embed_params, replace=False)
        else:
            indxs = np.arange(num_embed_params)
        self.embed_params = [self.params[i] for i in indxs]
        self.params = [p for i, p in enumerate(self.params) if i not in indxs]
        self.param_vals = [p for i, p in enumerate(self.param_vals) if i not in indxs]
        self.embed_param_indxs_orig = indxs

    def _replace_for_as_params_all_embed(self):
        # This method is called before the circuit is doubled
        self.orig_param_list = tuple(self.params)

        indxs = np.random.choice(self.num_params, size=self.num_params, replace=False)
        self.embed_params = [self.params[i] for i in indxs]

        self.params = []
        self.param_vals = []
        self.embed_param_indxs_orig = indxs
    
    def _replace_for_as_params_iqp_embed(self):
        self.orig_param_list = tuple(self.params)
        # indxs = np.random.choice(self.num_params, size=self.num_params, replace=False)
        self.embed_params = self.params
        self.params = []
        self.param_vals = []
        self.embed_param_indxs_orig = range(self.num_params)

    def qke_transform(self):
        assert self.embed_params is not None, "QKE transform must occur after initial separation of parameters."
        reverse_applied_ops = list(reversed(self.applied_operators))
        num_embed_params = len(self.embed_params)
        new_embed_params = [get_param() for _ in range(num_embed_params)]
        self.embed_params.extend(new_embed_params)
        self.params.extend(self.params)
        self.param_vals.extend(self.param_vals)

        params = list(self.orig_param_list)    
        assert len(self.embed_param_indxs_orig) == len(new_embed_params), "Error in copying embedding params."
        for i, indx in enumerate(self.embed_param_indxs_orig):
            params[indx] = new_embed_params[i]
        # self.params.extend(params)
        params = list(reversed(params))
        fin_hadarmard = True if self._qke_rand else False
        self._apply_ops_to_circ(reverse_applied_ops, fin_hadarmard=fin_hadarmard, 
                        params=params, hermitian=True)
        if self.embed == 'as_params_iqp':
            self._apply_ops_to_circ(reverse_applied_ops, fin_hadarmard=fin_hadarmard, 
                        params=params, hermitian=True)
            # self._apply_ops_to_circ(reverse_applied_ops, fin_hadarmard=fin_hadarmard, 
            #             params=params, hermitian=True)

    @staticmethod
    def repeat_params_map(states, len_example_state, num_params, qke_flag=True):
        num_states = len(states)
        len_of_input = len_example_state // 2 if qke_flag else len_example_state
        
        states = np.array(list(states))
        states = states.reshape(-1, len_of_input)
        states = np.tile(states, np.ceil(num_params / len_of_input).astype(int))[:, :num_params]
        return states.reshape(num_states, -1)
    
    @staticmethod
    def iqp_transform_data(states, n_qubits):
        state_list = []
        num_states = len(states)
        states = np.array(states)
        states = states.reshape((-1, len(states[0])//2))
        for state in states:
            s = np.outer(state, state).reshape(1, -1).squeeze()
            ss = np.concatenate([state, s])
            # ss = np.concatenate([ss,ss])
            state_list.append(ss)
            assert len(state_list[-1]) == (n_qubits + n_qubits**2), "Error in data transformation. {}!={}".format(len(state_list[-1]), (n_qubits + n_qubits**2))
        state_list = np.array(state_list).reshape((num_states, -1))
        return state_list

    def apply_mapping_matrix(self, states, len_of_input): # Not used anymore by as_params_all
        states = np.array(states)
        states = states.reshape(-1, len_of_input // 2)
        states = np.dot(states, self.mapping_matrix)
        states = states.reshape(-1, self.mapping_matrix.shape[1] * 2)
        return states

    def reset_device(self): # Used in multiprocessing
        self.device = reset_device(sample_num=self.pqc_sample_num)

    def reset_sample_num(self, sample_num):
        self.pqc_sample_num = sample_num
        self.reset_device()

    @classmethod
    def init_haar_random(cls, n_qubits, pqc_sample_num):
        """ This initiates a PQC with a Haar random unitary. """
        pqc = cls(n_qubits, num_params=0, pqc_sample_num=pqc_sample_num)
        rand_unitary = None
        print("This has not been implemented yet.")
        exit()
    
    @staticmethod
    def gen_mapping_matrix(in_size, out_size): # Not used anymore by as_params_all
        return 2 * np.random.rand(in_size, out_size) - 1

# ----------------------------------------------------------------------------------
# -------------------------- Useful functions --------------------------------------

def get_embed_transformed_states(states, embed, n_qubits, num_params, qke_flag=True):
    example_state = states.iloc[0] if isinstance(states, pd.Series) else states[0]
    if embed == 'as_params_all':
        return PQC.repeat_params_map(states, len_example_state=len(example_state), 
                                        num_params=num_params, qke_flag=qke_flag)
    elif embed == 'as_params_iqp':
        assert n_qubits == len(example_state) // 2, "Number of qubits must equal the dimesion: {}!={}".format(n_qubits, len(example_state)//2)
        return PQC.iqp_transform_data(states, n_qubits)
    else:
        return states

def exp_val_from_counts_list(counts_list, ob_q=None):
    """ ob_q determines the qubits from which the exp value is measured - others are ignored. """
    return [_exp_val_from_counts(c_dict, ob_q) for c_dict in counts_list]



def _exp_val_from_counts(counts_dict, ob_q=None):
    tot, exp = 0, 0
    for sample, counts in counts_dict.items():
        exp += counts * _exp_val_from_str(sample, ob_q) 
        tot += counts
    return exp / tot


def _exp_val_from_str(sample, ob_q=None):
    sample = "".join([sample[i] for i in ob_q]) if ob_q is not None else sample
    if sample.count('1') % 2:
        return -1
    return 1


def prob_0_from_counts_list(counts_list, sample_num):
    if sample_num == 'exact':
        return counts_list
    return [_prob_0_from_counts(c_dict, sample_num) for c_dict in counts_list]


def _prob_0_from_counts(counts_dict, sample_num):
    len_bit_str = len(next(iter(counts_dict.keys())))
    try:
        count_0 = counts_dict['0' * len_bit_str]
    except KeyError:
        count_0 = 0 
    return count_0 / sample_num


def reset_device(sample_num): # This reset is dependent on values set in device.py file
    return get_device(sample_num=sample_num)


def return_pauli_rot(pauli, param):
    if 'X' in pauli:
        rot = cirq.ops.Rx(rads=param)
    elif 'Y' in pauli:
        rot = cirq.ops.Ry(rads=param)
    else:
        rot = cirq.ops.Rz(rads=param)
    return rot


def return_seed():
    # This function is used as the seed for sampling the statevector - this is so that results can be reproduced
    return np.random.randint(9999) if SEED is None else SEED


def get_param():
    return sympy.Symbol(get_param_name())


def merge_dict(dict1, dict2):
    return {**dict2, **dict1}


def ret_rand_param_vals(num):
    return np.random.rand(num) * 2 * np.pi


def get_param_name():
    global _PARAM_COUNT
    _PARAM_COUNT += 1
    return "p{}".format(_PARAM_COUNT)


def weighting_from_count(counts):
    inv = (1 - np.array(counts) / sum(counts))
    if sum(inv) == 0:
        return np.array(counts) / sum(counts)
    return inv / sum(inv)


def get_pool_data():
    # Ensures code isn't slowed by constantly reopening potentially large file
    global KERNEL_POOL_FILE
    if KERNEL_POOL_FILE is None:
        KERNEL_POOL_FILE = unpickle_file(KERNEL_POOL_FILENAME)
    return KERNEL_POOL_FILE


def set_kernel_pool(filename):
    global KERNEL_POOL_FILENAME 

    KERNEL_POOL_FILENAME = filename
    get_pool_data()