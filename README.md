# Quantum-Random-Forest
The model is based off the paper Srikumar et al., "A kernel-based quantum random forest for improved classification", (2022). The code is intended for research purposes and the development of proof of concepts. For questions about the code and its use in the paper, please email maiyuren.s@gmail.com for clarification.

-----

## Required packages

Python=3.8.10 was used with the following libraries:

    - cirq==0.11.0
    - cirq-core==0.11.0
    - matplotlib==3.4.2
    - more-itertools==8.8.0
    - numpy==1.19.5
    - pandas==1.3.0
    - qiskit==0.27.0
    - scikit-learn==0.24.2
    - scipy==1.7.0
    - tqdm==4.61.1
    - tensorflow==2.4.1

-----

Example code is found in the example_notebook.ipynb file. Feel free to try with your own datasets.      

Note, only IQP and Efficient-Anzatz embeddings are available. Feel free to construct other embeddings from scratch and add to the pqc.py and split_function.py files.