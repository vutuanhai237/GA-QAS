import sys
sys.path.insert(0, '..')
import qtm.qcompilation, qtm.ansatz, qtm.state


class A:
    def __init__(self):
        self.t = 0
    def run(self):
        self.t = test_nqubit_tomography()
def test_nqubit_tomography():
    num_qubits = 3
    num_layers = 1
    compiler = qtm.qcompilation.QuantumCompilation(
        u = qtm.state.create_ghz_state(num_qubits).inverse(),
        vdagger = qtm.ansatz.Wchain_ZXZlayer_ansatz(num_qubits, num_layers),
        optimizer = 'adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=10, verbose = 1)
    return compiler.loss_values[-1]