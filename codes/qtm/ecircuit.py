import numpy as np
import random
import types
import qiskit
import qtm.evolution
import qtm.random_circuit
import qtm.state
import qtm.qcompilation
import qtm.ansatz
class ECircuit():
    def __init__(self, qc: qiskit.QuantumCircuit, task: types.FunctionType, pool) -> None:
        self.qc = qc
        self.task = task
        self.fitness = 0
        self.pool = pool
        self.status = 'uncompile'
        return
    def compile(self):
        self.fitness = self.task(self.qc)
        self.status = 'compiled'
        return
    def mutate(self, mute_func: types.FunctionType):
        self.qc = mute_func(self.qc, self.pool)
        return 
        
    def crossover(self, qc2, percent = 0.5):
        percent_sub1 = 1 - self.fitness / (self.fitness+ qc2.fitness)
        sub11, sub12 = qtm.evolution.divide_circuit(self.qc, percent_sub1)
        sub21, sub22 = qtm.evolution.divide_circuit(qc2.qc, 1 - percent_sub1)
        new_qc1 = ECircuit(qtm.evolution.compose_circuit([sub11, sub22]), self.task, self.pool)
        new_qc2 = ECircuit(qtm.evolution.compose_circuit([sub21,sub12]), self.task, self.pool)
        cr = qiskit.ClassicalRegister(self.qc.num_qubits, 'c')

        return new_qc1, new_qc2


   
