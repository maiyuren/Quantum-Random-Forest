import pickle
import os

def _free_filename(filename):
    indx = 1
    orig = filename
    while True:
        filename = orig.split('.pickle')[0] + '_ver{}.pickle'.format(indx)
        if not os.path.isfile(filename):
            break
        indx += 1
    return filename


def pickle_file(filename, ob, replace=False):

    if not replace:
        filename = _free_filename(filename)
    with open(filename, "wb") as f:
        pickle.dump(ob, f)


def unpickle_file(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print("\nSomething wrong with file:", filename)
        raise e
