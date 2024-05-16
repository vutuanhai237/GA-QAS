<<<<<<< Updated upstream
import sys
import matplotlib.pyplot as plt
import qiskit
from qsee.backend import constant
from qsee.compilation.qsp import QuantumStatePreparation, metric
from qsee.backend import utilities
from qsee.core import state
import json 
import numpy as np 
#purity
# 0 =  (0.24999999999999994+0j)
# 1 =  (0.5677667882613809+3.851859888774472e-34j)
# 2 =  (0.730855309056976-4.622231866529366e-33j)
# 3 =  (0.8580323099679779+1.5407439555097887e-33j)
# 4 =  (0.9322567793094236+3.0814879110195774e-33j)
# 5 =  (0.9692084935721368+1.5407439555097887e-33j)
# 6 =  (0.9863121420505355-7.703719777548943e-34j)
# 7 =  (0.9939753431498678+2.311115933264683e-33j)
# 8 =  (0.9973598385860678-6.162975822039155e-33j)
# 9 =  (0.9988452284134757-7.703719777548943e-34j)
# 10 =  (0.9994953416621428-2.311115933264683e-33j)

#fitness: fidelity of compilation, Gibbs - purity, fidelity
# ((0.5000249021981271-4.336808689942018e-18j), (0.573826708064223423+1.9389555183345926993e-10j))
# ((0.5892861344999469-3.350184712980209e-17j), (0.67126953117687258865+6.859320679662626195e-10j))
# ((0.7733364830385987-8.581460195222768e-17j), (0.76874728580091832714+3.0984539800907486113e-09j))
# ((0.9540115815427062+1.1622647289044608e-16j), (0.83493499729898033174-3.2632360934132222947e-09j))
# ((0.9951256541743505-3.054739620977909e-17j), (0.8477560066702951938-8.77352424760142735e-09j))
# ((0.9987405006185337+1.9895109865109006e-17j), (0.8536821398108414699+4.369205663975932591e-12j))
# ((0.9994088043314524+1.0511340062246965e-16j), (0.850886770402781753-3.839492532047049731e-09j))
# ((0.9998695408764084+1.1438332919722072e-16j), (0.85353234966501144443+5.4451499414148660983e-10j))
# ((0.9997756193415498-4.0766001685454967e-17j), (0.858092520933840322+4.6894143819219361226e-09j))
# ((0.9999576808661527-2.644098048149024e-17j), (0.8505097123128019451+2.936713229066721584e-09j))
# ((0.999998219297621+2.0355895788415346e-17j), (0.8529240633697925192+2.0467268469056892293e-09j))

#gibbs fidelity, layer = 7
# ((0.6986999295055334-9.8879238130678e-17j), (0.78459371599517374607+1.4428222886439616834e-08j))
# ((0.9997093245890737+6.830473686658678e-18j), (0.8354959170873575494-1.9144165980468024792e-09j))
# ((0.9999998291830037+2.721347452938616e-17j), (0.84354884873806858236+3.2098647069696960658e-09j))
# ((0.9999284313834365-6.938893903907228e-18j), (0.8546832966395262257+1.0707966700208225432e-09j))
# ((0.9999999974413647-2.710505431213761e-17j), (0.85337988406068510105+8.2073066483227058435e-09j))
# ((0.9999983784749937+3.9546274241408774e-17j), (0.8495112795203403483+3.9654447805658077745e-10j))
# ((0.9999706172571213+4.217546450968612e-17j), (0.8510073084241797231-5.743628702462662069e-09j))
# ((0.9999866811994067-1.235990476633475e-16j), (0.8530984305513011137-1.4896412911377081769e-09j))
# ((0.9998812670505904-4.575333167888829e-17j), (0.85578676610426927807+1.0660088735275419706e-09j))
# ((0.999969169586956+2.6454533008646308e-17j), (0.8499959793261145369+1.7887745033136928795e-09j))
# ((0.9999694125384133-1.6588293239028218e-17j), (0.84937357750341952595+5.7599714170803380418e-09j))

#later = 8
# ((0.9242438756712785-8.847089727481716e-17j), (0.5893313506183867345-3.7520503111385090815e-09j))
# ((0.8628300901968105-2.4936649967166602e-17j), (0.93942881123491262657-1.1396610844875817175e-12j))
# ((0.9417340503844556+6.5052130349130266e-18j), (0.9669171350440676632-2.9939961750815372832e-09j))
# ((0.9753590143401591+1.2674323396355547e-16j), (0.98333017016532096866+1.751848922693394788e-09j))
# ((0.9884912220843929-4.954803928258755e-17j), (0.9925687261649927817-2.805709868258501303e-09j))
# ((0.9962055935350422-1.3715157481941631e-17j), (0.9961352628265914694+6.028632238492330308e-09j))
# ((0.9984749996353464-5.2150124496552763e-17j), (0.99826426972649720537-7.0434424394292654787e-09j))
# ((0.9992102134278146+4.095573706563993e-17j), (0.999291312763725334+1.6896020784777531655e-09j))
# ((0.9995738422986945-8.903840934947754e-17j), (0.99969011439664416663+2.4310497670406745744e-09j))
# ((0.9999147316274328+1.2274015625483164e-16j), (0.9996991314758398306+1.576387299411171367e-13j))
# ((0.9999801069629396-5.2150124496552763e-17j), (0.9998960051748804895-2.7684607021102545693e-12j))

#numqubits = 3, layer=15
# ((0.2757556425012938-1.6086849734253672e-17j), (0.95821836302752090055+5.3577839290151926894e-10j))
# ((0.5883067918657396-8.009543549236664e-18j), (0.96403689416179774024-1.9501043380396562034e-09j))
# ((0.8231949631061202-4.358492733391728e-17j), (0.96639307181934315863+1.2885533870763827198e-10j))
# ((0.9182618862596461-5.659535340374333e-17j), (0.97145776484599909335+8.0767735966043850454e-09j))
# ((0.9249174499695896+2.6928871459108716e-17j), (0.9722266450142630703+6.286424708321062704e-11j))
# ((0.9322715216777203+3.946495907847236e-17j), (0.9733634988753738018-4.1044275052543139904e-09j))
# ((0.9275616742412618-1.1492543028346347e-17j), (0.97430832113354480517-5.8154172044334197672e-12j))
# ((0.9288224366950248-3.5344990823027445e-17j), (0.97281902615790321123+5.745407858744653016e-09j))
# ((0.9365965417821167-7.07712968089913e-17j), (0.9711848763675471827+8.226372145696864447e-11j))
# ((0.9289602959969359-7.177418381854039e-17j), (0.9721398908498527086+7.303306072961468617e-11j))
# ((0.9351201646986054+1.1881585261020247e-16j), (0.9652502495831434895+1.2997735265005010164e-09j))

#ground truth
# 0 =  (0.2853173109469464+2.322871840057251e-17j)
# 1 =  (0.5674352475823294+6.61443614366776e-18j)
# 2 =  (0.8146042338137478+1.4203733339855864e-33j)
# 3 =  (0.889848245264326-3.562970397116386e-33j)
# 4 =  (0.9088084559220145-3.562970397116386e-33j)
# 5 =  (0.9133085069275994-7.2222372914521344e-34j)
# 6 =  (0.9143564476739869-3.129636159629258e-33j)
# 7 =  (0.9145992328543846-6.7407548053553255e-34j)
# 8 =  (0.9146554084802341-1.5888922041194696e-33j)
# 9 =  (0.9146684023494582+7.2222372914521344e-34j)
# 10 =  (0.9146714077131303+2.1666711874356403e-33j)

#COMPILATION FROM BEST CIRCUIT
# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_1_4qubits_compilation_fitness_gibbs_2024-01-25/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_1_4qubits_compilation_fitness_gibbs_2024-01-25/best_circuit_{beta_coeff}")
#     print("End")


# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_1_4qubits_compilation_fitness_gibbs_2024-01-17/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_1_4qubits_compilation_fitness_gibbs_2024-01-17/best_circuit_{beta_coeff}")
#     print("End")

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_4qubits_compilation_fitness_gibbs_2024-01-18/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_4qubits_compilation_fitness_gibbs_2024-01-18/best_circuit_{beta_coeff}")
#     print("End")

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_3_4qubits_compilation_fitness_gibbs_2024-01-19/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_3_4qubits_compilation_fitness_gibbs_2024-01-19/best_circuit_{beta_coeff}")
#     print("End")

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_4_4qubits_compilation_fitness_gibbs_2024-01-19/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_4_4qubits_compilation_fitness_gibbs_2024-01-19/best_circuit_{beta_coeff}")
#     print("End") 

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_5_4qubits_compilation_fitness_gibbs_2024-01-20/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_5_4qubits_compilation_fitness_gibbs_2024-01-20/best_circuit_{beta_coeff}")
#     print("End") 

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_6_4qubits_compilation_fitness_gibbs_2024-01-20/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_6_4qubits_compilation_fitness_gibbs_2024-01-20/best_circuit_{beta_coeff}")
#     print("End") 

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_7_4qubits_compilation_fitness_gibbs_2024-01-21/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_7_4qubits_compilation_fitness_gibbs_2024-01-21/best_circuit_{beta_coeff}")
#     print("End") 

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_8_4qubits_compilation_fitness_gibbs_2024-01-21/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_8_4qubits_compilation_fitness_gibbs_2024-01-21/best_circuit_{beta_coeff}")
#     print("End") 

# qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_9_4qubits_compilation_fitness_gibbs_2024-01-22/best_circuit')
# print(qc.depth(),qc.num_qubits)
# num_qubits = 2

# for beta_coeff in range(0,11):
#     print("Begin")
#     qsp = QuantumStatePreparation(
#         u=qc,
#         target_state= state.construct_tfd_state(num_qubits, beta = beta_coeff).inverse()
#     ).fit(100)

#     qsp.save(f"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_9_4qubits_compilation_fitness_gibbs_2024-01-22/best_circuit_{beta_coeff}")
#     print("End")

def load_data(num_qubits,filename):
    qc = utilities.load_circuit(filename+f'/best_circuit_{num_qubits}/u')
    tgt = utilities.load_circuit(filename+f'/best_circuit_{num_qubits}/vdagger')

    data = json.load(open(filename+f"/best_circuit_{num_qubits}/info.json"))

    gibbs_purity, gibbs_fidelity = metric.gibbs_metrics(u=qc,vdagger=tgt.inverse(),thetass=[data["thetas"]])

    return gibbs_purity, gibbs_fidelity

gibbs_purities0 = [] 
gibbs_fidelities0 = []

# gibbs_purities1 = [] 
# gibbs_fidelities1 = []

# gibbs_purities2 = [] 
# gibbs_fidelities2 = []

# gibbs_purities3 = [] 
# gibbs_fidelities3 = []

# gibbs_purities4 = [] 
# gibbs_fidelities4 = []

# gibbs_purities5 = [] 
# gibbs_fidelities5 = []

# gibbs_purities6 = [] 
# gibbs_fidelities6 = []

# gibbs_purities7 = [] 
# gibbs_fidelities7 = []

# gibbs_purities8 = [] 
# gibbs_fidelities8 = []

# gibbs_purities9 = [] 
# gibbs_fidelities9 = []

for i in range (11):
    gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_1_4qubits_compilation_fitness_gibbs_2024-01-25")
    
    gibbs_purities0.append(abs(gibbs_purity))
    gibbs_fidelities0.append(abs(gibbs_fidelity))


print(gibbs_purities0,gibbs_fidelities0)
#     gibbs_purities1.append(abs(gibbs_purity))
#     gibbs_fidelities1.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_4qubits_compilation_fitness_gibbs_2024-01-18")
    
#     gibbs_purities2.append(abs(gibbs_purity))
#     gibbs_fidelities2.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_3_4qubits_compilation_fitness_gibbs_2024-01-19")
    
#     gibbs_purities3.append(abs(gibbs_purity))
#     gibbs_fidelities3.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_4_4qubits_compilation_fitness_gibbs_2024-01-19")
    
#     gibbs_purities4.append(abs(gibbs_purity))
#     gibbs_fidelities4.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_5_4qubits_compilation_fitness_gibbs_2024-01-20")
    
#     gibbs_purities5.append(abs(gibbs_purity))
#     gibbs_fidelities5.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_6_4qubits_compilation_fitness_gibbs_2024-01-20")
    
#     gibbs_purities6.append(abs(gibbs_purity))
#     gibbs_fidelities6.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_7_4qubits_compilation_fitness_gibbs_2024-01-21")
    
#     gibbs_purities7.append(abs(gibbs_purity))
#     gibbs_fidelities7.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_8_4qubits_compilation_fitness_gibbs_2024-01-21")
    
#     gibbs_purities8.append(abs(gibbs_purity))
#     gibbs_fidelities8.append(abs(gibbs_fidelity))

#     gibbs_purity, gibbs_fidelity = load_data(i,"/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_9_4qubits_compilation_fitness_gibbs_2024-01-22")
    
#     gibbs_purities9.append(abs(gibbs_purity))
#     gibbs_fidelities9.append(abs(gibbs_fidelity))

# print(gibbs_purities0)
# print(gibbs_fidelities0)
# print()


# print(gibbs_purities1)
# print(gibbs_fidelities1)
# print()

# print(gibbs_purities2)
# print(gibbs_fidelities2)
# print()

# print(gibbs_purities3)
# print(gibbs_fidelities3)
# print()

# print(gibbs_purities4)
# print(gibbs_fidelities4)
# print()

# print(gibbs_purities5)
# print(gibbs_fidelities5)
# print()

# print(gibbs_purities6)
# print(gibbs_fidelities6)
# print()

# print(gibbs_purities7)
# print(gibbs_fidelities7)
# print()

# print(gibbs_purities8)
# print(gibbs_fidelities8)

# print()
# print(gibbs_purities9)
# print(gibbs_fidelities9)


# gibbs_purities0 = [0.2500969687399561, 0.5772369448138033, 0.7669964920670586, 0.8767791952727861, 0.9484926118192514, 0.9785635729619069, 0.9886252008452465, 0.9964777017731079, 0.9991228741559122, 0.9999215080671643, 0.9987868421425388]
# gibbs_fidelities0 = [0.99995156309322949517, 0.9834310092148971713, 0.9894906206624379044, 0.9944989706889123676, 0.9973979305253908856, 0.99876152720202467095, 0.9994701752968944808, 0.9997028131666193473, 0.99979422134874555705, 0.99962203525082964084, 0.99990278627642018776]
# gibbs_purities1 = [0.500079683608921, 0.7926921261518977, 0.9358302209338764, 0.9759830558081111, 0.9884400134821367, 0.995706936406962, 0.9989837447164673, 0.9990045550120403, 0.9993558398002695, 0.999607606278964, 0.9997447701107592]
# gibbs_fidelities1 = [0.7070927015779671096, 0.9548208440745864226, 0.96889458120826560394, 0.98317535581233648264, 0.99225171961397131554, 0.99634479881897687614, 0.99789996754296849584, 0.99911711407919567287, 0.9996861098709958129, 0.9998724514450891693, 0.999931861858679926]

# mean_line = np.average(np.stack([gibbs_purities0,gibbs_purities1]),axis=0)
# min_line = np.min(np.stack([gibbs_purities0,gibbs_purities1]),axis=0)
# max_line = np.max(np.stack([gibbs_purities0,gibbs_purities1]),axis=0)
=======
import sys, qiskit
import qiskit.quantum_info
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
import qoop
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.core import ansatz, state, random_circuit
from qoop.backend import constant, utilities
from qoop.evolution import crossover, mutate, selection, threshold
from qoop.evolution.environment import EEnvironment, EEnvironmentMetadata
import pickle
import json
from qoop.compilation.qsp import QuantumStatePreparation, metric, QuantumCompilation
import glob 

filename = ""

def multiple_compile(num_qubitss,qcs,betas):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(bypass_compile,num_qubitss,qcs,betas)
    return results

def bypass_compile(num_qubits,qc,beta):
    return compiliation(num_qubits,qc,beta)

def compiliation(num_qubits,qc,beta):
    constant.PSI = state.construct_tfd_state(num_qubits, beta = beta,return_vector=True)
    qsp = QuantumCompilation(
            u=qc,
            vdagger = qiskit.QuantumCircuit(num_qubits*2,num_qubits*2)
        ).fast_fit(num_steps=500)
    gibbs_purity, gibbs_fidelity = metric.gibbs_metrics(u=qc,vdagger=constant.PSI.conj(),thetass=[qsp.thetas])
    # qsp.save(filename+f"/best_circuit_{beta}")
    return gibbs_purity, gibbs_fidelity 

def create_2qubit_circ(thetas):
    # from qiskit.circuit import ParameterVector
    # thetas = ParameterVector('theta',21*4)
    qc = qiskit.QuantumCircuit(4)
    qc.rx(thetas[0],2)
    qc.cry(thetas[1],1,3)
    qc.rx(thetas[2],0)
    qc.cx(1,2)
    qc.crx(thetas[3],0,3)
    qc.h(0)
    qc.h(1)
    qc.rx(thetas[4],2)
    qc.rx(thetas[5],3)
    qc.crz(thetas[6],0,1)
    qc.cx(2,3)
    qc.crx(thetas[7],1,3)
    qc.rx(thetas[8],2)
    qc.rz(thetas[9],0)
    qc.rz(thetas[10],1)
    qc.ry(thetas[11],3)
    qc.rx(thetas[12],0)
    qc.rz(thetas[13],2)
    qc.cry(thetas[14],0,3)
    qc.h(0)
    qc.cry(thetas[15],1,2)
    qc.crz(thetas[16],3,1)
    qc.ry(thetas[17],2)
    qc.rx(thetas[18],1)
    qc.ry(thetas[19],0)
    qc.cry(thetas[20],3,2)

    return qc 

def add_extra_rot_gates(qc,numqubit,thetas):
    for i in range(numqubit):
        qc.rx(thetas[i],i)
    for i in range(numqubit):
        qc.rz(thetas[i+numqubit],i)
    for i in range(numqubit):
        qc.rx(thetas[i+2*numqubit],i)
    return qc 

def create_block_3_qubits(thetas):
    qc = qiskit.QuantumCircuit(6)
    # extra_params = 18*3
    # orig_params = 21*3
    from qiskit.circuit import ParameterVector
    
    # thetas = ParameterVector('theta',orig_params)
    
    qc2b = create_2qubit_circ(thetas[:21])
    qc = qc.compose(qc2b,[0,1,3,4])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params:orig_params+18])

    
    qc2b = create_2qubit_circ(thetas[21:2*21])
    qc = qc.compose(qc2b,[1,2,4,5])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+18:orig_params+2*18])
    
    qc2b = create_2qubit_circ(thetas[2*21:3*21])
    qc = qc.compose(qc2b,[0,2,3,5])
    return qc
 
def train_bestc_circuit(filename):
    # qc2qbs = utilities.load_circuit(filename+"/best_circuit")

    thetas = None 

    num_qubits = 4


    # qc = qiskit.QuantumCircuit(6)
    # extra_params = 18*3
    # orig_params = 21*3
    # from qiskit.circuit import ParameterVector
    
    # thetas = ParameterVector('theta',orig_params+extra_params)
    # qc2b = create_2qubit_circ(thetas[:21])
    # qc = qc.compose(qc2b,[0,1,3,4])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params:orig_params+18])

    
    # qc2b = create_2qubit_circ(thetas[21:2*21])
    # qc = qc.compose(qc2b,[1,2,4,5])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+18:orig_params+2*18])
    
    # qc2b = create_2qubit_circ(thetas[2*21:3*21])
    # qc = qc.compose(qc2b,[0,2,3,5])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+2*18:orig_params+3*18])


    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+2*18:orig_params+3*18])
    qc = qiskit.QuantumCircuit(8)
    orig_param = 3*4*21 
    from qiskit.circuit import ParameterVector
    thetas = ParameterVector('theta',orig_param)

    qc_3_1 = create_block_3_qubits(thetas[:3*21]) 
    qc = qc.compose(qc_3_1,[0,1,2,4,5,6])

    qc_3_2 = create_block_3_qubits(thetas[3*21:2*3*21])
    qc = qc.compose(qc_3_2,[1,2,3,5,6,7])
    
    qc_3_3 = create_block_3_qubits(thetas[2*3*21:3*3*21])
    qc = qc.compose(qc_3_3,[0,2,3,4,6,7])

    qc_3_4 = create_block_3_qubits(thetas[3*3*21:4*3*21])
    qc = qc.compose(qc_3_4,[0,1,3,4,5,7])
    
    print(qc)
    print(qc.depth())
    gibbs_purities, gibbs_fidelities = [], []
    betas = np.linspace(0,10,100)
    res = multiple_compile([num_qubits]*len(betas),[qc]*len(betas),betas)
    
    gibbs_purities = []
    gibbs_fidelities = []
    for r in res:
        gibbs_purities.append(r[0])
        gibbs_fidelities.append(r[1])

    print(gibbs_fidelities)
    print(gibbs_purities)

    
    print("End")
        


train_bestc_circuit("/home/viet/GA-QAS/4qubits_compilation_fitness_gibbs_2024-04-26")
# beta = np.linspace(0,10,100)
# res = []
# for b in beta:
#     psi = state.construct_tfd_state(4, beta = b,return_vector=True)
#     psi = np.expand_dims(psi,axis=0)
#     rho = psi.conj().T*psi
#     gibbs_sigma = qiskit.quantum_info.partial_trace(rho, [0, 1,2,3])
#     res.append(np.trace(np.linalg.matrix_power(gibbs_sigma, 2)))
# print(res)

# (0.06250000000000003+0j), (0.06771330733853156+0j), (0.08445778558545108+0j), (0.11456343218667628+0j), (0.15765933412703767+0j), (0.20981673068187412+0j), (0.2649830167857983+0j), (0.31756284053729633+0j), (0.36403546053872876+0j), (0.4030895373946566+0j), (0.4349479800932326+0j), (0.460601620552052+0j), (0.4812691655903131+0j), (0.49810764389455864+0j), (0.5120954726225081+0j), (0.5240094269394089+0j), (0.5344433839148754+0j), (0.5438406310160007+0j), (0.5525266686433912+0j), (0.560737525816196+0j), (0.5686424275420752+0j), (0.5763612235136013+0j), (0.5839774879324829+0j), (0.5915482404586769+0j), (0.5991111106929768+0j), (0.6066896021391113+0j), (0.6142969569696184+0j), (0.6219389957303081+0j), (0.6296162074631213+0j), (0.6373252916051917+0j), (0.6450602983268987+0j), (0.652813474017669+0j), (0.6605758895967077+0j), (0.6683379082789895+0j), (0.6760895341741908+0j), (0.6838206720335611+0j), (0.6915213204231846+0j), (0.6991817147522166+0j), (0.7067924323166603+0j), (0.7143444683979827+0j), (0.7218292901674944+0j), (0.7292388734652644+0j), (0.7365657262826217+0j), (0.7438029018607891+0j), (0.750944003638573+0j), (0.7579831837760187+0j), (0.7649151366027169+0j), (0.7717350880549607+0j), (0.7784387819508554+0j), (0.7850224637883971+0j), (0.7914828626252763+0j), (0.7978171715009147+0j), (0.8040230267838987+0j), (0.810098486766136+0j), (0.8160420097749849+0j), (0.8218524320334105+0j), (0.8275289454637618+0j), (0.8330710756016055+0j), (0.838478659761015+0j), (0.8437518255709486+0j), (0.8488909699833618+0j), (0.8538967388368413+0j), (0.8587700070446855+0j), (0.8635118594631043+0j), (0.8681235724833787+0j), (0.8726065963813201+0j), (0.8769625384480062+0j), (0.8811931469174619+0j), (0.885300295699609+0j), (0.8892859699203756+0j), (0.8931522522651753+0j), (0.8969013101171119+0j), (0.9005353834770249+0j), (0.9040567736489264+0j), (0.9074678326713579+0j), (0.9107709534727144+0j), (0.9139685607265756+0j), (0.917063102381479+0j), (0.9200570418383907+0j), (0.922952850748232+0j), (0.9257530024013085+0j), (0.928459965680146+0j), (0.9310761995472371+0j), (0.9336041480393039+0j), (0.9360462357400288+0j), (0.9384048637036941+0j), (0.9406824058027461+0j), (0.9428812054730031+0j), (0.9450035728310567+0j), (0.9470517821392348+0j), (0.9490280695944096+0j), (0.9509346314179159+0j), (0.9527736222247851+0j), (0.9545471536515071+0j), (0.9562572932225472+0j), (0.9579060634367937+0j), (0.9594954410561773+0j), (0.9610273565795919+0j), (0.9625036938862663+0j), (0.9639262900336394+0j)]

    # qc = qiskit.QuantumCircuit(6)
    # extra_params = 18*3
    # orig_params = 21*3
    # from qiskit.circuit import ParameterVector
    
    # thetas = ParameterVector('theta',orig_params)
    
    # qc2b = create_2qubit_circ(thetas[:21])
    # qc = qc.compose(qc2b,[0,1,3,4])
    # # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params:orig_params+18])

    
    # qc2b = create_2qubit_circ(thetas[21:2*21])
    # qc = qc.compose(qc2b,[1,2,4,5])
    # # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+18:orig_params+2*18])
    
    # qc2b = create_2qubit_circ(thetas[2*21:3*21])
    # qc = qc.compose(qc2b,[0,2,3,5])
    # # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+2*18:orig_params+3*18])
>>>>>>> Stashed changes
