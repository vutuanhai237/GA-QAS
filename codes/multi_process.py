import qiskit
import sys
sys.path.insert(0, '..')
import qtm.qcompilation
import matplotlib.pyplot as plt
import qtm.qsp
import qtm.ansatz, qtm.state
import itertools
import os
from multiprocessing import Pool


folder_path = '../experiments/qsp_1/'

# Get a list of all files and directories in the folder
all_items = os.listdir(folder_path)

# Filter out only the file names
file_names = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

list_num_qubits = [3,4,5,6,7,8,9,10]
list_num_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list_u = ['g2', 'g2gn', 'g2gnw']
list_v = ['ghz', 'W', 'haar']

# Generate all combinations of the lists
combinations = list(itertools.product(list_num_qubits, list_num_layers, list_u, list_v))

# Your existing functions and code
def f(num_qubits, num_layers, anszat, vdagger_input):
    optimizer = 'adam'

    if anszat == 'g2':
        u_input = qtm.ansatz.g2(num_qubits, num_layers) 
        ansatz_input = qtm.ansatz.g2
    if anszat == 'g2gn':
        u_input = qtm.ansatz.g2gn(num_qubits, num_layers) 
        ansatz_input = qtm.ansatz.g2gn
    if anszat == 'g2gnw':
        u_input = qtm.ansatz.g2gnw(num_qubits, num_layers) 
        ansatz_input = qtm.ansatz.g2gnw
        
    if vdagger_input == 'ghz':
            vdagger = qtm.state.create_ghz_state(num_qubits).inverse()
    if vdagger_input == 'W':
            vdagger = qtm.state.create_w_state(num_qubits).inverse()
    if vdagger_input == 'AME':
            vdagger = qtm.state.create_ame_state(num_qubits).inverse()   
    if vdagger_input == 'haar':
            vdagger = qtm.state.create_haar_state_inverse(num_qubits) 
    
    compiler = qtm.qcompilation.QuantumCompilation(
        u = u_input,
        vdagger = vdagger,
        optimizer = optimizer,
        loss_func = 'loss_fubini_study'
    )
    compiler.fit(num_steps = 100, verbose = 1)
    qspobj = qtm.qsp.QuantumStatePreparation.load_from_compiler(
    compiler = compiler, ansatz = ansatz_input)
    qspobj.save(state = vdagger_input, file_name='../experiments/qsp_1/')

    return 

def process_combination(combination):
    i, j, k, l = combination
    name = f"{l}_{k}_{i}_{j}.qspobj"

    if name not in file_names:
        print(name)
        f(i, j, k, l)

if __name__ == "__main__":

    # Create a Pool with the number of processes you want to use
    num_processes = 4  # Adjust this based on your machine's capabilities
    with Pool(num_processes) as pool:
        # Use pool.map to process combinations in parallel
        pool.map(process_combination, combinations)

    print("Done!")