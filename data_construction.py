import os
import numpy as np
from scipy.stats import norm
from qiskit.quantum_info import random_statevector
import pandas as pd
import random
import itertools
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tqdm
from pathlib import Path
import tensorflow as tf
from pickle_functions import pickle_file, unpickle_file

# -------------------------------------------------------------------------------------------
# --------------------------------- Training sets -------------------------------------------


def scikit_clas_datasets(train_prop=0.75, ret_embed=False, dataset='iris', encode_half_rot=True, **kwargs):
    """ This function obtains the iris data set and normalises values for embedding. """
    if dataset == 'iris':
        data = datasets.load_iris()
    elif dataset == 'breast_cancer':
        data = datasets.load_breast_cancer()
    elif dataset == 'digits':
        data = datasets.load_digits()
    elif dataset == 'wine':
        data = datasets.load_wine()
    else:
        data = None; print("Error in loading."); exit()
    
    X = data.data
    if 'X_dim' in kwargs:
        DATASET_DIM = kwargs['X_dim']
        X, _ = truncate_x(X, None, n_components=DATASET_DIM)
        print(f'New datapoint dimension:', len(X[0]))
    
    X = q_embed_normalise(X, encode_half_rot=encode_half_rot)

    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)
    train_data = pd.DataFrame(zip(X_train, y_train), columns=['X', 'y'])
    test_data = pd.DataFrame(zip(X_test, y_test), columns=['X', 'y'])

    if ret_embed:
        embed = {i:v for i, v in enumerate(data.target_names)}
        return train_data, test_data, embed
    return train_data, test_data


def data_preprocessing(X, y, train_prop=0.75, X_dim=None, encode_half_rot=True):
    if X_dim is not None:
        X, _ = truncate_x(X, None, n_components=X_dim)
        print(f'New datapoint dimension:', len(X[0]))
    
    X = q_embed_normalise(X, encode_half_rot=encode_half_rot)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)
    train_data = pd.DataFrame(zip(X_train, y_train), columns=['X', 'y'])
    test_data = pd.DataFrame(zip(X_test, y_test), columns=['X', 'y'])
    return train_data, test_data


def local_datasets(dataset, train_prop=0.75, ret_embed=False, folder='', **kwargs):
    """ This function downloads the datasets that are present in the datasets folder. """
    if folder == '':
        folder = str(Path(__file__).absolute()).split('data_construction.py')[0]
    if dataset == 'higgs':
        return higgs_dataset(train_prop=train_prop, ret_embed=ret_embed, folder=folder, **kwargs)
    elif dataset == 'abalone':
        return abalone_dataset(train_prop=train_prop, ret_embed=ret_embed, folder=folder, **kwargs)
    elif dataset == 'ionosphere':
        return ionosphere_dataset(train_prop=train_prop, ret_embed=ret_embed, folder=folder, **kwargs)
    elif dataset == 'heart':
        return heart_dataset(train_prop=train_prop, ret_embed=ret_embed, folder=folder, **kwargs)
    else:
        print(f"No such dataset found. Given {dataset}.")


def all_datasets(dataset, train_prop=0.75, ret_embed=False, folder='', qrf_optimal=False, **kwargs):
    """ All datasets can be accessed through this function. """
    if 'load_relabelled' in kwargs and kwargs['load_relabelled']:
        assert 'embed' in kwargs and 'num_params' in kwargs, "Error in loading."
        if check_dataset_relabelled(dataset, kwargs['num_params'], kwargs['embed'], qrf_optimal=qrf_optimal):
            data_df = load_dataset_relabelled(dataset, kwargs['num_params'], kwargs['embed'], qrf_optimal=qrf_optimal)
            if train_prop == -1:
                print("Returning raw data file as DataFrame.")
                return data_df
            X_train, X_test, y_train, y_test = train_test_split(data_df.X, data_df.y, train_size=train_prop)
            if ret_embed:
                return pd.DataFrame(zip(X_train, y_train), columns=['X', 'y']), \
                             pd.DataFrame(zip(X_test, y_test), columns=['X', 'y']), None
            else:
                return pd.DataFrame(zip(X_train, y_train), columns=['X', 'y']), \
                                pd.DataFrame(zip(X_test, y_test), columns=['X', 'y'])
        else:
            raise NotRelabelledError

    if dataset in ['iris', 'breast_cancer', 'digits', 'wine']:
        return scikit_clas_datasets(train_prop=train_prop, ret_embed=ret_embed, dataset=dataset, **kwargs)
    elif dataset == 'fashion_mnist':
        return fashion_mnist_dataset(train_prop=train_prop, ret_embed=ret_embed, folder=folder, **kwargs)
    elif dataset == 'co_linear':
        return co_linear_dataset(train_prop=train_prop, ret_embed=ret_embed, folder=folder)
    else:
        return local_datasets(dataset=dataset, train_prop=train_prop, ret_embed=ret_embed, folder=folder, **kwargs)


def abalone_dataset(train_prop=0.75, ret_embed=False, folder='', tot_num_data='all', **kwargs):
    data = pd.read_csv(folder + 'datasets/abalone/abalone.csv')
    label_col = data.Rings
    y = (label_col >= 11).astype(int)
    X = data.iloc[:, 1:-1] # we leave out the sex categorical column
    X = np.array(X)
    
    NUM = tot_num_data if tot_num_data != 'all' else len(X)
    if tot_num_data != 'all':
        random_indices = random.sample(range(len(X)), NUM)
        X, y = np.array(X[random_indices]), y[random_indices]
        y = np.array(y * 1)

    if 'X_dim' in kwargs:
        DATASET_DIM = kwargs['X_dim']
        X, _ = truncate_x(X, None, n_components=DATASET_DIM)
        print(f'New datapoint dimension:', len(X[0]))

    X = q_embed_normalise(X, encode_half_rot=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)
    train_data = pd.DataFrame(zip(X_train, y_train), columns=['X', 'y'])
    test_data = pd.DataFrame(zip(X_test, y_test), columns=['X', 'y'])

    if ret_embed:
        embed = {0:'< 11', 1:">=11"}
        return train_data, test_data, embed
    return train_data, test_data


def ionosphere_dataset(train_prop=0.75, ret_embed=False, folder='', **kwargs):
    data = pd.read_csv(folder + 'datasets/ionosphere/ionosphere.csv')

    labels = data.label.unique()
    embed = {l:i for i, l in enumerate(labels)}
    y = [embed[l] for l in data.label] 

    X = data.iloc[:, :-1]
    X = np.array(X)
    
    if 'X_dim' in kwargs:
        DATASET_DIM = kwargs['X_dim']
        X, _ = truncate_x(X, None, n_components=DATASET_DIM)
        print(f'New datapoint dimension:', len(X[0]))
    
    X = q_embed_normalise(X, encode_half_rot=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)
    train_data = pd.DataFrame(zip(X_train, y_train), columns=['X', 'y'])
    test_data = pd.DataFrame(zip(X_test, y_test), columns=['X', 'y'])

    if ret_embed:
        return train_data, test_data, embed
    return train_data, test_data


def heart_dataset(train_prop=0.75, ret_embed=False, folder='', **kwargs):
    data = pd.read_csv(folder + 'datasets/heart/heart.csv')
    y = list(data.target)

    X = data.iloc[:, :-1]
    X = np.array(X)
   
    if 'X_dim' in kwargs:
        DATASET_DIM = kwargs['X_dim']
        X, _ = truncate_x(X, None, n_components=DATASET_DIM)
        print(f'New datapoint dimension:', len(X[0]))
    
    X = q_embed_normalise(X, encode_half_rot=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)
    train_data = pd.DataFrame(zip(X_train, y_train), columns=['X', 'y'])
    test_data = pd.DataFrame(zip(X_test, y_test), columns=['X', 'y'])

    if ret_embed:
        labels = data.target.unique()
        embed = labels 
        return train_data, test_data, embed
    return train_data, test_data



def fashion_mnist_dataset(train_prop=0.75, ret_embed=False, folder='', num_classes=2, tot_num_data=300, **kwargs):
    """ The data preparation here is idential to the work of Huang et al. 2021. """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    CLASS_SELECTION_ORDER = [0, 3, 5, 1, 2, 4, 6, 7, 8, 9]
    
    if num_classes == 2: 
        x_train, y_train = filter_03(x_train, y_train)
        x_test, y_test = filter_03(x_test, y_test)
    elif num_classes == 'all':
        pass
    elif num_classes > 2:
        x_train, y_train = filter_classes(x_train, y_train, CLASS_SELECTION_ORDER[:num_classes])
        x_test, y_test = filter_classes(x_test, y_test, CLASS_SELECTION_ORDER[:num_classes])
    else: 
        print("Cannot obtain {} number of classes from fashion_mnist".format(num_classes)); exit()
    
    if 'X_dim' in kwargs:
        DATASET_DIM = kwargs['X_dim']
    else:
        DATASET_DIM = 5
    
    x_train, x_test = truncate_x(x_train, x_test, n_components=DATASET_DIM)
    print(f'New datapoint dimension:', len(x_train[0]))

    # We only look at the training set and split later 
    x_train = q_embed_normalise(np.array(x_train), encode_half_rot=True)

    NUM = tot_num_data if tot_num_data != 'all' else len(x_train)
    random_indices = random.sample(range(len(x_train)), NUM)
    X, y = np.array(x_train[random_indices]), y_train[random_indices]
    y = np.array(y * 1)

    if train_prop == -1:
        return pd.DataFrame(zip(X, y), columns=['X', 'y'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)

    train_data = pd.DataFrame(zip(X_train, y_train), columns=['X', 'y'])
    test_data = pd.DataFrame(zip(X_test, y_test), columns=['X', 'y'])


    if ret_embed:
        labels = np.unique(y_train)
        embed = labels 
        return train_data, test_data, embed

    return train_data, test_data


# ----------------------------------------------------------------------------------
# -------------------------- Useful functions --------------------------------------

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *( (s+b+br) * np.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return np.sqrt(radicand)


def AMS_metric(predictions, testing_df, embedding):
    """  Prints the AMS metric value to screen. """
    
    signal = 0.0
    background = 0.0

    assert len(predictions) == len(testing_df), "Number of predictions and labels should match."

    for i in tqdm.tqdm(range(len(predictions))):
        row = testing_df.iloc[i]
        if predictions[i] == embedding['s']: # only events predicted to be signal are scored
            if row.y == embedding['s']:
                signal += float(row.weight)
            elif row.y == embedding['b']:
                background += float(row.weight)
     
    print('signal = {}, background = {}'.format(signal, background))
    print('AMS = ' + str(AMS(signal, background)))


def normalise(a):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return a / np.sqrt(np.vdot(a, a))


def gaussian_distr(length, std_dev=None, mean=0, range_distr=None):
    """ This function returns a Gaussian distribution state, given mu and sigma. """
    if not range_distr:
        range_distr = [-length / 2, length / 2]

    std_dev = length // 4 if std_dev is None else std_dev
    x = np.linspace(range_distr[0], range_distr[1], length)

    return normalise(np.array(norm.pdf(x, mean, std_dev)))


def create_gaussian_distributions(N, list_mean=None, list_std=None, complex=False):

    if not isinstance(list_mean, np.ndarray) and not isinstance(list_mean, list):
        list_mean = np.arange(-N / 3.0, N / 3.0, N / 20.0)
    if not isinstance(list_std, np.ndarray) and not isinstance(list_std, list):
        list_std = np.linspace(N / 50.0, N / 3.0, 10)

    yield len(list_std)*len(list_mean)

    for mean in list_mean:
        for std in list_std:
            d = gaussian_distr(N, std_dev=std, mean=mean, range_distr=[-N / 2, N / 2])
            if complex:
                d = d.numpy() * random_phases(N)
            yield d, mean, std


def random_phases(N):
    out = np.random.random(2*N) * (np.random.randint(2, size=2*N) * 2 - 1)
    return normalise_complex(out.view(np.complex128))


def normalise_complex(a):
    assert isinstance(a, np.ndarray)

    return a/np.abs(a)


def ret_gaussian_distributions(N, list_mean=None, list_std=None):
    """ Returns a set of Gaussian distributions that can be used for training. """
    distributions = []
    std_deviations = []
    means = []
    distr_gen = create_gaussian_distributions(N, list_mean, list_std)
    size_distr = next(distr_gen)
    for d, m, s in distr_gen:
        distributions.append(d)
        means.append(m)
        std_deviations.append(s)

    return distributions, means, std_deviations


def return_state(numQubits, stateType, **kwargs):
    """ This function returns various types of states as statevectors. """
    target = None
    if stateType == 'wState':
        target = np.zeros(2**numQubits, dtype=np.complex128)
        bit = 1
        for i in range(2**numQubits):
            if i == bit:
                target[i] = 1 + 0j
                bit *= 2
            else:
                target[i] = 0 + 0j

        target /= np.sqrt(np.vdot(target, target))
    elif stateType == 'randomState':
        target = random_statevector(2**numQubits).data
    elif stateType == 'gaussianState':
        try:
            mu, std = kwargs['mu'], kwargs['std']
        except KeyError:
            mu, std = 0, None
        mu, sigma = (2**numQubits)/2, (2**numQubits)/3
        target = gaussian_distr(2**numQubits, std_dev=std, mean=mu)
    else:
        print("No such state type.")
        exit(2345)
    return target


def state_discr_a_states(num):
    a_list = np.random.random(num)
    return np.array([normalise([np.sqrt(1-a), 0, a, 0]) for a in a_list]), np.array([0 for _ in range(num)])


def state_discr_b_states(num):
    num_pos = num // 2
    out_states_pos = np.array([normalise([0, np.sqrt(1/2), np.sqrt(1/2), 0])] * num_pos)
    out_states_neg = np.array([normalise([0, -np.sqrt(1/2), np.sqrt(1/2), 0])] * (num - num_pos))
    return np.concatenate([out_states_pos, out_states_neg]), np.array([1 for _ in range(num)])


def states_to_df_clas(states, class_labels):
    """ This is for classification. """
    assert len(states) == len(class_labels), "The number of states must equal the number of class labels."
    embed = {c:i for i, c in enumerate(set(class_labels))}
    y_embed = [embed[c] for c in class_labels]
    return pd.DataFrame(zip(states, class_labels, y_embed), columns=['X', 'labels', 'y']), \
           {v:k for v,k in embed.items()}


def q_embed_normalise(X, encode_half_rot=True):
    
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    assert len(X.shape) == 2, "Data must be a 2D array. A feature vector for each instance."

    n_col = X.shape[1]

    if encode_half_rot:
        prefactor = 0.5
    else:
        prefactor = 1

    for col in range(n_col):
        v = X[:, col]
        scale = v.max() - v.min()
        if scale == 0:
            X[:, col] = prefactor * (2 * np.pi * (v - v.min()) - np.pi)
        else:
            X[:, col] = prefactor * ((2 * np.pi * (v - v.min()) / scale) - np.pi)

    return X


def ret_random_state(n_qubits):
    return return_state(n_qubits, 'randomState')


def ret_bi_sep_state(n_qubits):
    sep_q = np.random.choice(np.arange(1, n_qubits))
    f_reg = ret_random_state(sep_q)
    s_reg = ret_random_state(n_qubits - sep_q)

    return normalise(np.kron(f_reg, s_reg))


def bloch_angles(state):
    assert len(state) == 2, "Bloch angles can only be obtained for single qubit."
    phi = (np.angle(state[1]) - np.angle(state[0])) % (2 * np.pi)
    theta = 2 * np.arccos(np.abs(state[0]))
    return theta, phi


def label_fn_gen(poly_deg, coef):
    assert poly_deg + 2 == len(coef), "Polynomial degree must match size of coefficients list."
    coef = np.array(coef)
    def label_fn(theta, phi):
        theta_ar = np.array([theta ** i for i in range(poly_deg, -1, -1)] + [1])
        phi_ar = np.array([phi ** i for i in range(poly_deg, -1, -1)] + [1])
        return int((theta_ar * phi_ar * coef).sum() > 0)
    
    return label_fn


def label_custom_fn_gen(type_fn=1):

    if type_fn == 1:
        def label_fn(theta, phi):
            if theta < np.pi / 3 :
                return 1
            elif theta > 2 * np.pi / 3:
                return 2
            return 0
    elif type_fn == 2:
        def label_fn(theta, phi):
            r = np.pi * np.sin(6 * phi) / 12  + np.pi / 3
            if theta < r:
                return 1
            elif theta > np.pi - r:
                return 2
            return 0 

    return label_fn


def ret_polar_from_cartesian(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    if x == 0:
        phi = np.pi / 2
    else:
        phi = np.arctan(y/x)
    if z == 0:
        theta = np.pi / 2 
    else:
        theta = np.arctan(np.sqrt(x**2 + y**2) / z)
    return r, theta, phi


def ret_state_from_cartesian(x, y, z):
    _, theta, phi = ret_polar_from_cartesian(x, y, z)
    return np.array([np.cos(theta), np.exp(1j* phi) * np.sin(theta)])


def fibonacci_sphere(samples=1):
    """ Code by Fab von Bellingshausen. """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points


def all_same(iterable):
    el = iterable[0]
    for i in iterable:
        if el != i:
            return False
    return True 


def mod_exp(g, p, indx):
    # indx can be a ndarray
    y = g ** indx
    return np.mod(y, p)


def check_dataset_relabelled(dataset, num_params, embed, qrf_optimal=False, kernel=False):
    filename = relabelled_filename(dataset, num_params, embed, qrf_optimal=qrf_optimal, kernel=kernel)
    return os.path.isfile(filename) 


def load_dataset_relabelled(dataset, num_params, embed, qrf_optimal=False):
    filename = relabelled_filename(dataset, num_params, embed, qrf_optimal=qrf_optimal)
    ob = unpickle_file(filename)
    print("Relabelled dataset loaded. (dataset, num_params, embed)=({},{},{})".format(dataset, num_params, embed))
    print("Filename:", filename)
    return ob


def save_dataset_relabelled(data_df, dataset, num_params, embed, qrf_optimal=False):
    filename = relabelled_filename(dataset, num_params, embed, qrf_optimal=qrf_optimal)
    print(filename)
    pickle_file(filename, data_df, replace=True)


def save_kernel_for_dataset_relabelled(data_df, kernel, dataset, num_params, embed, qrf_optimal=False):
    filename = relabelled_filename(dataset, num_params, embed, kernel=True, qrf_optimal=qrf_optimal)
    print(filename)
    ob = {'data_df': data_df, "kernel":kernel}
    pickle_file(filename, ob, replace=True)


def relabelled_filename(dataset, num_params, embed, kernel=False, qrf_optimal=False):
    if isinstance(embed, list):
        if all_same(embed):
            embed = embed[0]
        else:
            embed = ','.join(embed)
    if isinstance(num_params, list):
        if all_same(num_params):
            num_params = num_params[0]
        else:
            num_params = ','.join([str(s) for s in num_params])
    name = "{}{}__num_params-{}__embed-{}{}.pickle".format("" if not kernel else "kernel__", 
                                                         dataset, num_params, embed, 
                                                         "__qrf_opt" if qrf_optimal and not kernel else "")
    path = str(Path(__file__).absolute()).split('data_construction.py')[0]
    return "{}/datasets/relabelled_datasets/{}{}".format(path, 
                                                        "" if not kernel else "kernels/",
                                                        name)

class NotRelabelledError(Exception):
    pass

# ----------------------------------------------------------------------------------
# ------------------ Functions for Fashion MNIST prep ------------------------------

""" 
The following section of code was obtained from:
https://www.tensorflow.org/quantum/tutorials/quantum_data#2_relabeling_and_computing_pqk_features 
"""

def truncate_x(x_train, x_test, n_components=10):
  """Perform PCA on image dataset keeping the top `n_components` components."""
  n_points_train = tf.gather(tf.shape(x_train), 0)
  # Flatten to 1D
  x_train = tf.reshape(x_train, [n_points_train, -1])
  # Normalize.
  feature_mean = tf.reduce_mean(x_train, axis=0)
  x_train_normalized = x_train - feature_mean
  # Truncate.
  e_values, e_vectors = tf.linalg.eigh(
      tf.einsum('ji,jk->ik', x_train_normalized, x_train_normalized))

  if x_test is not None:
    n_points_test = tf.gather(tf.shape(x_test), 0)
    x_test = tf.reshape(x_test, [n_points_test, -1])
    x_test_normalized = x_test - feature_mean
    return tf.einsum('ij,jk->ik', x_train_normalized, e_vectors[:,-n_components:]), \
        tf.einsum('ij,jk->ik', x_test_normalized, e_vectors[:, -n_components:])
  else:
    return tf.einsum('ij,jk->ik', x_train_normalized, e_vectors[:,-n_components:]), None


def filter_03(x, y):
    keep = (y == 0) | (y == 3)
    x, y = x[keep], y[keep]
    y = y == 0
    return x,y


def filter_classes(x, y, class_list):

    keep = (y == class_list[0])
    for c in class_list[1:]: 
        keep = keep | (y == c) 

    x, y = x[keep], y[keep]

    y_out = np.zeros(len(y)).astype(int)
    for i, c in enumerate(class_list):
        y_out[y == c] = i

    return x, y_out


def get_stilted_dataset(K1, K2, lambdav=1.1, qrf_optimal=False):
  """ Prepare new labels that maximize geometric distance between kernels.
      Here we are given two kernel matrices. This function has been edited from the original. 
      K1 should be the quantum kernel. """
  
  if qrf_optimal:
      return qrf_optim_y(K1)

  S, V = get_spectrum(K1)
  S_2, V_2 = get_spectrum(K2)

  S_diag = tf.linalg.diag(S ** 0.5)
  S_2_diag = tf.linalg.diag(S_2 / (S_2 + lambdav) ** 2)
  scaling = S_diag @ tf.transpose(V) @ \
            V_2 @ S_2_diag @ tf.transpose(V_2) @ \
            V @ S_diag

  # Generate new lables using the largest eigenvector.
  _, vecs = tf.linalg.eig(scaling)
  new_labels = tf.math.real(
      tf.einsum('ij,j->i', tf.cast(V @ S_diag, tf.complex128), vecs[-1])).numpy()
  # Create new labels and add some small amount of noise.
  final_y = new_labels > np.median(new_labels)
  noisy_y = (final_y ^ (np.random.uniform(size=final_y.shape) > 0.95))
  return noisy_y.astype(int)


def get_spectrum(kernel_matrix, gamma=1.0):
  """Compute the eigenvalues and eigenvectors of the kernel matrix."""
  S, V = tf.linalg.eigh(kernel_matrix)
  S = tf.math.abs(S)
  return S, V


# ----------------------------------------------------------------------------------


def qrf_optim_y(K):
    """ Following A4 variation of QRF relabelling. """
    indxs = np.random.choice(np.arange(len(K)), size=2, replace=False)
    x1_y = K[:, indxs[0]]
    x2_y = K[:, indxs[1]]

    p = x2_y - x1_y
    
    # print("\n\n p")
    # print(p)
    # print("\n\n")

    q1 = np.quantile(p, 0.25)
    q2 = np.quantile(p, 0.5)
    q3 = np.quantile(p, 0.75)

    return (p >= q3).astype(int) + ((p >= q1) * (p < q2)).astype(int)
