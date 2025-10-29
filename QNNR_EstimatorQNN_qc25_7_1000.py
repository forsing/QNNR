# QNNR (Quantum Neural Network Regressor)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.visualization import circuit_drawer

from IPython.display import display
from IPython.display import clear_output


from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


"""
svih 4502 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 28.10.2025.
"""

# 1. Učitaj loto podatke
df = pd.read_csv("/data/loto7_4502_k85.csv", header=None)


###################################
print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37
"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:


"""
####################################


# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

# 3. Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# 4. Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)

# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4484, Broj pozicija: 7
"""


print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""
Prvih 5 mapiranih kombinacija:

    0   1   2   3   4   5   6
0   4  12  12  13  23  24  27
1   1   1  10  14  14  17  30
2  12  15  15  16  16  20  32
3  16  18  20  22  30  30  31
4   2   2   5   7  24  26  30
"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""
Zadnjih 5 mapiranih kombinacija:


"""




# Parametri
num_qubits = 5          # 5 qubita po poziciji
num_layers = 2          # Dubina varijacionog sloja
num_positions = 5       # 6 pozicija (brojeva) u loto kombinaciji

def encode_position(value):
    """
    Sigurno enkoduje 'value' u QuantumCircuit sa tacno num_qubits qubita.
    Ako value zahteva vise bitova od num_qubits, koristi se LSB (zadnjih num_qubits bitova),
    i ispisuje se upozorenje.
    """
    # osiguraj int
    v = int(value)
    bin_full = format(v, 'b')  # pravi binarni bez vodećih nula
    if len(bin_full) > num_qubits:
        # upozorenje: vrednost ne staje u broj qubita; koristimo zadnjih num_qubits bita (LSB)
        print(f"Upozorenje: value={v} zahteva {len(bin_full)} bitova, a num_qubits={num_qubits}. Koristim zadnjih {num_qubits} bita.")
        bin_repr = bin_full[-num_qubits:]
    else:
        bin_repr = bin_full.zfill(num_qubits)

    qc = QuantumCircuit(num_qubits)
    # reversed da bi LSB išao na qubit 0 (ako želiš suprotno, ukloni reversed)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == '1':
            qc.x(i)
    return qc




# Varijacioni sloj: Ry rotacije + CNOT lanac
def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc

# QCBM ansambl: slojevi varijacionih blokova
def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer * num_qubits
        end = (layer + 1) * num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc

# Kompletan QCBM za svih 7 pozicija
def full_qcbm(params_list, values):
    total_qubits = num_qubits * num_positions
    qc = QuantumCircuit(total_qubits)

    for pos in range(num_positions):
        start_q = pos * num_qubits
        end_q = start_q + num_qubits

        # Enkoduj vrednost za poziciju
        qc_enc = encode_position(values[pos])
        qc.compose(qc_enc, qubits=range(start_q, end_q), inplace=True)

        # Dodaj varijacioni ansambl
        qc_var = qcbm_ansatz(params_list[pos])
        qc.compose(qc_var, qubits=range(start_q, end_q), inplace=True)

    # Dodaj merenja za svih 30 qubita
    qc.measure_all()

    return qc


test_values = [3,15,21,25,28,29,34]
np.random.seed(39)
params_list = [np.random.uniform(0, 2*np.pi, num_layers * num_qubits) for _ in range(num_positions)]

# Generiši QCBM za svih 6 pozicija
full_circuit = full_qcbm(params_list, test_values)



# Prikaz celog kruga u 'mpl' formatu
full_circuit.draw('mpl')
# plt.show()

# fold=40 prelama linije tako da veliki krug stane na ekran.
full_circuit.draw('mpl', fold=40)
# plt.show()


# The only valid choices are 
# text, latex, latex_source, and mpl


# Kompaktni prikaz kola
print("\nKompaktni prikaz kvantnog kola (text):\n")
# print(full_circuit.draw('text'))
"""
Kompaktni prikaz kvantnog kola (text):


"""


# display(full_circuit.draw())     
display(full_circuit.draw("mpl"))
# plt.show()


circuit_drawer(full_circuit, output='latex', style={"backgroundcolor": "#EEEEEE"})
# plt.show()


# import tinytex
# pip install tinycio
# pip install torchvision
# tinytex.install()



"""
# Sačuvaj kao PDF
img1 = full_circuit.draw('latex')
img1.save("/data/qc30_5_1.pdf")


# Sačuvaj kao sliku u latex formatu jpg
img2 = full_circuit.draw('latex')
img2.save("/data/qc30_5_2.jpg")


# Sačuvaj kao sliku u latex formatu png
img3 = full_circuit.draw('latex')
img3.save("/data/qc30_5_3.png")


# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/data/qc30_5_4.jpg")

# Sačuvaj kao sliku u matplotlib formatu png
img5 = full_circuit.draw('mpl', fold=40)
img5.savefig("/data/qc30_5_5.png")
"""




# Sačuvaj kao sliku u matplotlib formatu jpg
img4 = full_circuit.draw('mpl', fold=40)
img4.savefig("/KvantniRegresor/2QNNR/QNNR_Est_qc25_7_4.jpg")



###############################################



# QNNR (Quantum Neural Network Regressor)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from tqdm import tqdm
import random


from qiskit_machine_learning.circuit.library import QNNCircuit


from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


from qiskit_machine_learning.optimizers import GradientDescent

from qiskit_aer.primitives import Sampler as AerSampler

from qiskit.circuit.library import TwoLocal


from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN


from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.circuit import ParameterVector




# QNNR (Quantum Neural Network Regressor)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit_aer import AerSimulator
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from tqdm import tqdm
import random

from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

from qiskit_machine_learning.optimizers import GradientDescent

from qiskit_aer.primitives import Sampler as AerSampler

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import matplotlib.pyplot as plt




# =========================
# 2. Koristimo samo zadnjih N=1000 za test
# =========================
N = 1000  #4500
df = df.tail(N).reset_index(drop=True)





X = df.iloc[:, :-1].values  # prvih 5 brojeva
y_full = df.values          # svi 6 brojeva (5+1)

# Skaliranje
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X).astype(np.float64)
print()
print("X_scaled.shape[0]")
print(X_scaled.shape[0])
print()
"""
X_scaled.shape[0]
4484
"""

print()
print("len(X_scaled)")
print(len(X_scaled))
print()
"""
len(X_scaled)
4484
"""




# Define a custom interpret function that calculates the parity of the bitstring
def parity(x):
    return f"{bin(x)}".count("1") % 2


# =========================
# Treniranje i predikcija po brojevima
# =========================


predicted_combination = []

for i in range(7):  # 5 brojeva + 2 dodatna broja
    print(f"\n--- Treniranje QNN regresora za broj {i+1} ---")
    y = y_full[:, i].astype(np.float64)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
    print()
    print("y_scaled.shape[0]")
    print(y_scaled.shape[0])
    print()
    """
    y_scaled.shape[0]
    4484
    """

    print()
    print("len(y_scaled)")
    print(len(y_scaled))
    print()
    """
    len(y_scaled)
    4484
    """

    
    # SamplerQNN sa lokalnim AerSimulator-om
    backend = AerSimulator()


    num_qubits = X_scaled.shape[1]
    print()
    print("num_qubits = X_scaled.shape[1]")
    print(num_qubits)
    print()
    """
    num_qubits = X_scaled.shape[1]
    6   
    """



    # Example 2: Explicitly specifying the feature map and ansatz
    # Create a feature map and an ansatz separately
    # feature_map = ZZFeatureMap(feature_dimension=num_qubits)
    # ansatz = RealAmplitudes(num_qubits=num_qubits)

    # Compose the feature map and ansatz manually (otherwise done within QNNCircuit)
    # qc = QuantumCircuit(num_qubits)
    # full_circuit.compose(feature_map, inplace=True)
    # full_circuit.compose(ansatz, inplace=True)

    # Example 1: Using the QNNCircuit class
    # QNNCircuit automatically combines a feature map and an ansatz into a single circuit
    
    sampler = Sampler()

    
    gradient = GradientDescent()  # param-shift rule


    # 1. Kreiraj feature map sa parametrima za ulaz
    feature_map = ZZFeatureMap(feature_dimension=num_qubits)

    # 2. Kreiraj ansatz sa težinskim parametrima
    # ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2)


    
    ansatz = TwoLocal(num_qubits=num_qubits,
                  rotation_blocks='ry',
                  entanglement_blocks='cz',
                  entanglement='full',
                  reps=2)
    


    params = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)

    # trening sa progress barom
    pbar = tqdm(total=len(X_scaled), desc="QNNR EstimatorQNN trening")
    # pbar = tqdm(total=num_samples, desc=f"Broj {i+1}")

    theta = params.copy()
    for _ in range(len(X_scaled)):
        # theta = optimizer(lambda th: cost(th), theta)
        pbar.update(1)
    pbar.close()




    # 3. Spoji ih u jedan parametarski krug
    full_circuit_map = feature_map.compose(ansatz)
    # full_circuit = feature_map.compose(feature_map)


    


    # full_circuit.compose(ansatz.assign_parameters(theta), inplace=True)
    # full_circuit.measure(range(num_qubits), range(num_qubits))  # merenja



    regression_estimator_qnn = EstimatorQNN(
        circuit=full_circuit_map,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        gradient=gradient
    )






    # from qiskit_machine_learning.neural_networks import SamplerQNN, circuit_parity
    
    


    # NeuralNetworkRegressor
    
    # optimizer = COBYLA(maxiter=100) 

    
    optimizer = COBYLA(maxiter=len(X_scaled))



    # priprema target distribucije
    def numbers_to_bitstring(row, n_qubits):
        # mapira sve brojeve u jedan bitstring dužine n_qubits
        return ''.join([format(int(v)-1, '06b') for v in row])


    target_counts = {}
    for row in y_full:
        bitstr = numbers_to_bitstring(row, num_qubits)
        target_counts[bitstr] = target_counts.get(bitstr, 0) + 1
    for k in target_counts:
        target_counts[k] /= len(y_full)


    


    # Primer: kreiramo progress bar
    total_iters = len(X_scaled)  # postavi u skladu sa optimizer.maxiter
    pbar = tqdm(total=total_iters, desc=f"Broj {i+1}")

    # Definišemo callback koji ažurira progress bar
    def progress_callback(weights, loss):
        pbar.update(1)
    

    regressor = NeuralNetworkRegressor(
        neural_network=regression_estimator_qnn,
        loss='squared_error',
        optimizer=optimizer,
        callback=progress_callback
    )


    # Fit sa progres bar
    # Fit poziva callback automatski
    regressor.fit(X_scaled, y_scaled)
    pbar.close()



    # Predikcija sledećeg broja
    last_scaled = scaler_X.transform([X[-1]]).astype(np.float64)
    pred_scaled = regressor.predict(last_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).round().astype(int)[0][0]

    predicted_combination.append(int(pred))
    print(f"Predikcija za broj {i+1}: {pred}")

print()
print("\n=== Predviđena sledeća loto kombinacija (5+2) ===")
print(" ".join(str(num) for num in predicted_combination))
print()
"""
4500
4500
=== Predviđena sledeća loto kombinacija (5+2) ===
2 10 x x x 24 20



8 13 16 

4500/4500 [20:22:09<00:00, 16.30s/it]

2166/4500 [9:11:18<9:54:03, 15.27s/it

"""



"""
=== Qiskit Version Table ===
Software                       Version        
---------------------------------------------
qiskit                         1.4.4          
qiskit_machine_learning        0.8.3          

=== System Information ===
Python version                 3.11.13        
OS                             Darwin         
Time                           Tue Sep 09 18:11:49 2025 CEST
"""



