#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


# QNNR (Quantum Neural Network Regressor)
# Regression with an EstimatorQNN


"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


# QNNR_EstimatorQNN_qc25_7_1000_v2 — poboljšana varijanta.
# CSV Num1–Num7 ili 7 kolona; podrazumevano loto7hh_4584_k23.csv; --n-tail (podrazumevano 1000).
# COBYLA iz qiskit_algorithms (izbegava ImportError u qiskit_machine_learning.optimizers).
# Shim: ako u aktivnom Pythonu nedostaju simboli u qiskit.primitives / .utils (mešane verzije,
# drugi venv), dodaju se zamene pre uvoza qiskit_machine_learning. Preporuka: qiskit 1.4.4 + qml 0.8.3 (README).
# Bez IPython display; StatevectorEstimator; opcioni --save-circuit; --maxiter 0 = max(100, len(X)).

"""
svih 4584 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 20.03.2026.
"""

import argparse
import random
import sys
import warnings
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import TwoLocal, ZZFeatureMap


def _shim_qiskit_primitives_for_qml() -> None:
    """Ako nedostaju BaseEstimator, BaseSampler, _circuit_key, init_observable u primitives — dodaj zamene (bez pretpostavke verzije)."""
    import qiskit.primitives as qp
    import qiskit.primitives.utils as qpu

    def _stub(name: str) -> type:
        return type(name, (ABC,), {})

    for _n in (
        "BaseEstimator",
        "BaseEstimatorV1",
        "Estimator",
        "EstimatorResult",
        "BaseSampler",
        "BaseSamplerV1",
        "Sampler",
        "SamplerResult",
    ):
        if not hasattr(qp, _n):
            setattr(qp, _n, _stub(_n))

    if not hasattr(qpu, "_circuit_key"):

        def _circuit_key(circuit):  # type: ignore[no-untyped-def]
            """Stub: u starom Qiskitu keš ključ za kolo; ovde strukturalni otisak."""
            try:
                bits: list[tuple[str, tuple]] = []
                for inst in getattr(circuit, "data", ()) or ():
                    op = inst.operation
                    name = getattr(op, "name", type(op).__name__)
                    qi = tuple(getattr(b, "_index", hash(b)) for b in inst.qubits)
                    bits.append((name, qi))
                return (getattr(circuit, "num_qubits", None), tuple(bits))
            except Exception:
                return ("_circuit_key_fallback", id(circuit))

        qpu._circuit_key = _circuit_key  # type: ignore[attr-defined]

    if not hasattr(qpu, "init_observable"):

        def init_observable(observable):  # type: ignore[no-untyped-def]
            """Stub: stari qiskit.primitives.utils.init_observable → SparsePauliOp."""
            from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

            if isinstance(observable, SparsePauliOp):
                return observable
            if isinstance(observable, Pauli):
                return SparsePauliOp(observable)
            if isinstance(observable, Operator):
                return SparsePauliOp.from_operator(observable)
            if isinstance(observable, str):
                return SparsePauliOp(Pauli(observable))
            try:
                return SparsePauliOp.from_operator(observable)
            except Exception:
                return SparsePauliOp(observable)

        qpu.init_observable = init_observable  # type: ignore[attr-defined]


_shim_qiskit_primitives_for_qml()

from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 39
_DEFAULT_CSV = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4584_k23.csv")
_DEFAULT_OUT = Path(__file__).resolve().parent / "QNNR_EstimatorQNN_qc25_7_1000_v2_out"

min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]


def _get_cobyla(maxiter: int):
    try:
        from qiskit_algorithms.optimizers import COBYLA

        return COBYLA(maxiter=maxiter)
    except ImportError as e:
        raise ImportError(
            "Instaliraj qiskit-algorithms (pip install qiskit-algorithms) "
            "ili uskladi Qiskit; COBYLA iz qiskit_machine_learning.optimizers "
            "može da pukne zbog BaseSampler."
        ) from e


def _estimator_gradient(estimator):
    try:
        from qiskit_machine_learning.gradients import LinCombEstimatorGradient

        return LinCombEstimatorGradient(estimator)
    except Exception:
        return None


def load_seven_numbers(csv_path: Path) -> np.ndarray:
    df_head = pd.read_csv(csv_path, nrows=1, encoding="utf-8")
    if "Num1" in df_head.columns:
        cols = [f"Num{i}" for i in range(1, 8)]
        df = pd.read_csv(csv_path, encoding="utf-8")
        return df[cols].to_numpy(dtype=np.int64)
    df = pd.read_csv(csv_path, header=None, encoding="utf-8")
    if df.shape[1] < 7:
        raise ValueError("Potrebno najmanje 7 kolona brojeva")
    return df.iloc[:, :7].to_numpy(dtype=np.int64)


def map_to_indexed_range(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float64).copy()
    for i in range(7):
        out[:, i] = arr[:, i] - min_val[i]
        hi = max_val[i] - min_val[i]
        if not np.logical_and(out[:, i] >= 0, out[:, i] <= hi).all():
            raise ValueError(f"Kolona {i}: van dozvoljenog opsega")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="QNNR EstimatorQNN v2")
    ap.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--n-tail",
        type=int,
        default=1000,
        help="Poslednjih N redova (kao N=1000 u polaznom)",
    )
    ap.add_argument(
        "--maxiter",
        type=int,
        default=0,
        help="COBYLA maxiter; 0 = max(100, broj uzoraka)",
    )
    ap.add_argument(
        "--save-circuit",
        action="store_true",
        help="Sačuvaj MPL šemu prvog feature_map+ansatz u --out-dir",
    )
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--no-progress", action="store_true", help="Isključi tqdm")
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"Greška: nema CSV: {args.csv}", file=sys.stderr)
        sys.exit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)
    algorithm_globals.random_seed = args.seed

    data = load_seven_numbers(args.csv)
    _ = map_to_indexed_range(data)

    n_rows = data.shape[0]
    n_tail = max(1, min(args.n_tail, n_rows))
    tail = data[-n_tail:].copy()

    X = tail[:, :-1].astype(np.float64)
    y_full = tail.astype(np.float64)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X).astype(np.float64)

    n_samples = len(X_scaled)
    maxiter = args.maxiter if args.maxiter > 0 else max(100, n_samples)

    print()
    print(f"Učitano redova (ceo CSV): {n_rows}; za trening: poslednjih {n_tail}")
    print(f"X: {X_scaled.shape}, y: {y_full.shape}; COBYLA maxiter={maxiter}")
    print()

    estimator = StatevectorEstimator()
    n_q_feat = X_scaled.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=n_q_feat)
    ansatz = TwoLocal(
        num_qubits=n_q_feat,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="full",
        reps=2,
    )
    full_circuit_map = feature_map.compose(ansatz)

    if args.save_circuit:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        fig_mpl = full_circuit_map.draw("mpl", fold=40)
        path = args.out_dir / "qnnr_estimator_circuit.png"
        fig_mpl.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig_mpl)
        print(f"[plot] Sačuvano: {path}")

    predicted_combination: list[int] = []

    for i in range(7):
        print(f"\n--- Treniranje QNN regresora za broj {i + 1} ---")
        y = y_full[:, i]
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        optimizer = _get_cobyla(maxiter=maxiter)
        grad = _estimator_gradient(estimator)

        regression_estimator_qnn = EstimatorQNN(
            circuit=full_circuit_map,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
            gradient=grad,
        )

        pbar = tqdm(total=n_samples, desc=f"Broj {i + 1}", disable=args.no_progress)

        def progress_callback(_weights, _loss) -> None:
            pbar.update(1)

        regressor = NeuralNetworkRegressor(
            neural_network=regression_estimator_qnn,
            loss="squared_error",
            optimizer=optimizer,
            callback=progress_callback,
        )

        regressor.fit(X_scaled, y_scaled)
        pbar.close()

        last_scaled = scaler_X.transform([X[-1]]).astype(np.float64)
        pred_scaled = regressor.predict(last_scaled)
        pred = int(
            scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).round().astype(int)[0, 0]
        )
        predicted_combination.append(pred)
        print(f"Predikcija za broj {i + 1}: {pred}")

    print()
    print("=== Predviđena sledeća loto kombinacija (5+2) ===")
    print(" ".join(str(n) for n in predicted_combination))
    print()


if __name__ == "__main__":
    main()


"""
python3 QNNR_EstimatorQNN_qc25_7_1000_v2.py
python3 QNNR_EstimatorQNN_qc25_7_1000_v2.py --n-tail 100 --maxiter 200 --save-circuit
"""


"""

Učitano redova (ceo CSV): 4584; za trening: poslednjih 100
X: (100, 6), y: (100, 7); COBYLA maxiter=200


--- Treniranje QNN regresora za broj 1 ---
Broj 1:   0%|                         | 0/100 [00:52<?, ?it/s]
Predikcija za broj 1: x

--- Treniranje QNN regresora za broj 2 ---
Broj 2:   0%|                         | 0/100 [01:02<?, ?it/s]
Predikcija za broj 2: y

--- Treniranje QNN regresora za broj 3 ---
Broj 3:   0%|                         | 0/100 [00:59<?, ?it/s]
Predikcija za broj 3: 4

--- Treniranje QNN regresora za broj 4 ---
Broj 4:   0%|                         | 0/100 [01:15<?, ?it/s]
Predikcija za broj 4: 5

--- Treniranje QNN regresora za broj 5 ---
Broj 5:   0%|                         | 0/100 [00:55<?, ?it/s]
Predikcija za broj 5: 10

--- Treniranje QNN regresora za broj 6 ---
Broj 6:   0%|                         | 0/100 [00:55<?, ?it/s]
Predikcija za broj 6: z

--- Treniranje QNN regresora za broj 7 ---
Broj 7:   0%|                         | 0/100 [00:51<?, ?it/s]
Predikcija za broj 7: 25

=== Predviđena sledeća loto kombinacija (5+2) ===
x y 4 5 10 z 25
"""



"""
Za EstimatorQNN umesto GradientDescent kao „gradienta“ koristi se LinCombEstimatorGradient(estimator) ako postoji, inače None — u skladu sa QML API-jem.

--maxiter za COBYLA (iz qiskit_algorithms):

--maxiter 0 (podrazumevano) → koristi se max(100, broj uzoraka) (isti red veličine kao u polaznom gde je bilo maxiter=len(X_scaled) za COBYLA).
--maxiter N sa N > 0 → tačno N iteracija.
"""
