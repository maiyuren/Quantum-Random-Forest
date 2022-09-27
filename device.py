
import cirq
import qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ, ClassicalRegister
import numpy as np
from qiskit.providers import QiskitBackendNotFoundError
from qiskit.compiler import transpile

provider = None
private_provider = None
private_provider_default = None
DEV = 'cirq'
DEVICE_MAP = None

BULK_JOBS = True
MAX_CIRC_PER_JOB = 800 

class QDevice:
    """ This object handles the logistics of how the circuits are run. This may be either
        simulated or on the real IBM devices. """

    def __init__(self, dev_type, sample_num, bulk_jobs=BULK_JOBS) -> None:
        
        self.dev_type = dev_type
        self.dev_name = None    # For when running on real devices
        if self.dev_type == 'cirq':
            self.dev = cirq.Simulator()
        else: # Goes into here assuming a name of a backend is given
            self.dev_name = self.dev_type
            self.dev_type = 'ibm'
            self.backend = self._get_backend()
        self.sample_num = sample_num
        assert sample_num != 'exact' or dev_type == 'cirq', "Exact results cannot be obtained from real device."
        
        # These functions are used for storing circuits which can be sent in bulk to IBM devices
        self.bulk_jobs = bulk_jobs
        self._stored_circs = []
        self._run_circs_flag = False # Flags whether circuits need to be run

        self.meas_indicies = None
        self.previous_ob_q = []
    
    def run(self, run_circ, **kwargs):
        """ This method returns a results object (different for cirq vs qiskit) given a cirq 
            circuit to run. The samples must be obtained subsequently in various methods. """

        if self.dev_type == 'cirq':
            # print(run_circ)
            return self.dev.simulate(run_circ, initial_state=kwargs.get('initial_state', None)) 
        elif self.dev_type == 'ibm':
            if self.bulk_jobs:
                circ = convert_cirq_to_qiskit(run_circ, qubits=kwargs['qubits'], 
                                                observed_q_label=kwargs['observed_q_label'])
                self.add_circ_to_run_list(circ)
                return None
            return run_cirq_circuit_on_qiskit(run_circ, qubits=kwargs['qubits'], 
                                              backend=self.backend, sample_num=self.sample_num, 
                                              observed_q_label=kwargs['observed_q_label'])
        else:
            print("Error in running circuit."); exit()
    
    def get_counts_from_results(self, results, observed_q_label, seed=None):
        if self.dev_type == 'cirq':
            if self.sample_num == 'exact':
                if self.previous_ob_q != list(observed_q_label):
                    self.previous_ob_q = list(observed_q_label)
                    self.meas_indicies =  [i for i in range(len(results.state_vector())) if check_meas_q_indx_0(i, observed_q_label)]
                return exact_prob_0(results.state_vector(), observed_q_label, indicies=self.meas_indicies)
            samples = cirq.sample_state_vector(results.state_vector(), indices=observed_q_label,
                                               repetitions=self.sample_num, seed=seed)
            return counts_from_samples(samples) 
        elif self.dev_type == 'ibm':
            if isinstance(results, qiskit.result.result.Result):
                return results.get_counts()
            assert self.bulk_jobs, "Error in obtaining counts from IBM device." 
            return None
        else:
            print("Error in obtaning samples from result."); exit()

    def _run_stored_circ(self):
        """ This method runs all the stored circuits in a bulk job sent to the IBM device. It then returns
            the counts for each as a list of dictionaries. """
        assert self.dev_type == 'ibm', "Only implemented for use on real devices."
        assert self._run_circs_flag and len(self._stored_circs) > 0, "Error in bulking circut jobs."
        
        num_circs = len(self._stored_circs)
        print("Running {} circuits on {}.".format(num_circs, self.dev_name))
        # print("Example circuit:"); print(self._stored_circs[0])
        job = execute(self._stored_circs, self.backend, shots=self.sample_num)

        self._stored_circs = []
        self._run_circs_flag = False 

        return job.result().get_counts() if num_circs != 1 else [job.result().get_counts()]
    
    def add_circ_to_run_list(self, circ):
        self._run_circs_flag = True
        self._stored_circs.append(circ)

    def _get_backend(self):
        try:
            return provider.get_backend(self.dev_name)
        except QiskitBackendNotFoundError:
            try:
                return private_provider.get_backend(self.dev_name)
            except QiskitBackendNotFoundError:
                return private_provider_default.get_backend(self.dev_name)

def set_device(dev_type, device_map=None):
    global DEV 
    DEV = dev_type
    
    global DEVICE_MAP
    if device_map is not None:
        DEVICE_MAP = device_map


def get_device(dev_type=None, sample_num=None):
    dev_type = DEV if dev_type is None else dev_type
    return QDevice(dev_type, sample_num=sample_num)

# ----------------------------------------------------------------------------------
# -------------------------- Useful functions --------------------------------------

def convert_cirq_to_qiskit(circuit, qubits, observed_q_label):
    meas_ob = cirq.measure(*[qubits[i] for i in observed_q_label])
    circuit.append(meas_ob)
    qasm_output = cirq.QasmOutput((circuit.all_operations()), qubits)
    qiskit_circ = QuantumCircuit().from_qasm_str(str(qasm_output))
    if DEVICE_MAP is not None:
        qiskit_circ = transpile(qiskit_circ, initial_layout=DEVICE_MAP)
    return qiskit_circ


def run_cirq_circuit_on_qiskit(circuit, qubits, backend, sample_num, observed_q_label):
    qasm_circuit = convert_cirq_to_qiskit(circuit, qubits, observed_q_label)
    # Execute the circuit qiskit backend
    job = execute(qasm_circuit, backend, shots=sample_num)
    # Grab results from the job
    return job.result()


def counts_from_samples(samples):
    unique, counts = np.unique(samples, return_counts=True, axis=0)
    return {"".join([str(i) for i in sample]): counts[i] for i, sample in enumerate(unique)}


def exact_prob_0(state_vec, ob_q_labels, indicies=None):
    N = len(state_vec)
    if N == 2**len(ob_q_labels):
        return np.absolute(state_vec[0])**2
    else:
        indicies = [i for i in range(N) if check_meas_q_indx_0(i, ob_q_labels)] if indicies is None else indicies
        return sum([np.absolute(state_vec[i])**2 for i in indicies])


def check_meas_q_indx_0(indx, ob_q_labels):
    """ This function returns True if the value of a specific statevector index must be included. """
    i = bin(indx)[2:]
    for q in ob_q_labels:
        if len(i) <= q:
            continue
        elif int(i[q]) == 0:
            return False
    return True


def load_ibm_cred():
    global provider
    global private_provider
    global private_provider_default
    provider = IBMQ.load_account()
    private_provider = IBMQ.get_provider(hub='ibm-q-melbourne', group='reservations')
    private_provider_default = IBMQ.get_provider(hub='ibm-q-melbourne', group='internal')
