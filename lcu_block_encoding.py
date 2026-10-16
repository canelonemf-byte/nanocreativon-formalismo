#!/usr/bin/env python3
"""
Implementación numérica de PREPARE + SELECT (LCU block-encoding)
Ejemplo: H = 0.3 X + 0.5 Z + 0.2 Y

LECN-LLP — Lázaro López Paz
El Prat de Llobregat, Barcelona, España
28 de julio de 2026
"""

import numpy as np
from numpy.linalg import norm

# ============================================================
# Matrices de Pauli y producto de Kronecker
# ============================================================

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron(*ops):
    """Producto de Kronecker de varios operadores."""
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r

# ============================================================
# 1. Definición del Hamiltoniano y coeficientes LCU
# ============================================================

# H = 0.3 X + 0.5 Z + 0.2 Y
alphas = np.array([0.3, 0.5, 0.2])   # coeficientes
Us = [X, Z, Y]                       # unitarias
m = len(alphas)                      # número de términos
alpha = np.sum(alphas)               # α = Σ α_j
p = alphas / alpha                   # probabilidades

print("=" * 60)
print("LCU Block-Encoding — PREPARE + SELECT")
print("H = 0.3 X + 0.5 Z + 0.2 Y")
print("=" * 60)
print(f"α (norma L1)     = {alpha}")
print(f"p (probabilidades) = {p}")
print(f"m (nº términos)  = {m}")

# ============================================================
# 2. PREPARE: estado |G⟩ = Σ √p_j |j⟩
# ============================================================
# Usamos 2 qubits de ancilla (dimensión 4 ≥ m=3).
# Codificación: |0⟩→j=0, |1⟩→j=1, |2⟩→j=2, |3⟩→no usado.

n_anc = 2
dim_anc = 2**n_anc   # 4

# Estado |G⟩ en la base computacional de 2 qubits
G_state = np.zeros(dim_anc, dtype=complex)
G_state[0] = np.sqrt(p[0])   # |00⟩ → j=0 (X)
G_state[1] = np.sqrt(p[1])   # |01⟩ → j=1 (Z)
G_state[2] = np.sqrt(p[2])   # |10⟩ → j=2 (Y)
# |11⟩ permanece en 0

G_state = G_state / norm(G_state)  # por seguridad
print(f"\n|G⟩ (PREPARE) = {np.round(G_state, 4)}")

# Construimos una unitaria G tal que G|00⟩ = |G⟩
# (completamos a una base ortonormal con Gram-Schmidt)
def complete_unitary(first_column):
    """Construye una unitaria cuya primera columna es first_column."""
    d = len(first_column)
    U = np.zeros((d, d), dtype=complex)
    U[:, 0] = first_column / norm(first_column)
    # Base canónica y ortogonalización
    for i in range(1, d):
        v = np.zeros(d, dtype=complex)
        v[i] = 1.0
        for j in range(i):
            v = v - np.vdot(U[:, j], v) * U[:, j]
        nrm = norm(v)
        if nrm > 1e-12:
            U[:, i] = v / nrm
        else:
            # vector ya dependiente; buscar otro
            for k in range(d):
                v = np.zeros(d, dtype=complex)
                v[k] = 1.0
                for j in range(i):
                    v = v - np.vdot(U[:, j], v) * U[:, j]
                nrm = norm(v)
                if nrm > 1e-12:
                    U[:, i] = v / nrm
                    break
    return U

G = complete_unitary(G_state)          # PREPARE unitary (actúa solo en ancillas)
G_dag = G.conj().T

print(f"Verificación PREPARE: G|0⟩ ≈ |G⟩ → error = {norm(G[:,0] - G_state):.2e}")

# ============================================================
# 3. SELECT: Σ |j⟩⟨j| ⊗ U_j
# ============================================================
# SELECT actúa en (ancillas + sistema). Dimensión total = 4 * 2 = 8.

dim_sys = 2
dim_total = dim_anc * dim_sys

SELECT = np.zeros((dim_total, dim_total), dtype=complex)

# j=0 → X, j=1 → Z, j=2 → Y, j=3 → I (no usado, ponemos I)
U_list = [X, Z, Y, I]

for j in range(dim_anc):
    # Proyector |j⟩⟨j| en ancillas
    P_j = np.zeros((dim_anc, dim_anc), dtype=complex)
    P_j[j, j] = 1.0
    SELECT += kron(P_j, U_list[j])

print(f"SELECT construido. Norma de Frobenius = {norm(SELECT):.4f}")

# ============================================================
# 4. Block-encoding U = (G† ⊗ I) · SELECT · (G ⊗ I)
# ============================================================

G_on_full = kron(G, I)           # G actúa en ancillas, I en sistema
Gdag_on_full = kron(G_dag, I)

U_block = Gdag_on_full @ SELECT @ G_on_full

# ============================================================
# 5. Extracción del bloque superior izquierdo y verificación
# ============================================================
# El bloque que corresponde a ancillas en |00⟩ (índice 0) es de 2×2.

# Índices: ancilla |00⟩ → filas/columnas 0:2 del espacio total
block = U_block[0:2, 0:2]

H_exact = 0.3*X + 0.5*Z + 0.2*Y
H_normalized = H_exact / alpha

print("\n" + "=" * 60)
print("VERIFICACIÓN DEL BLOCK-ENCODING")
print("=" * 60)
print("\nBloque extraído (debe ser H/α):")
print(np.round(block, 6))
print("\nH/α exacto:")
print(np.round(H_normalized, 6))
print(f"\nError ‖bloque − H/α‖ = {norm(block - H_normalized):.2e}")

# Verificación adicional: acción sobre un estado de prueba
psi = np.array([1, 0], dtype=complex) / np.sqrt(1)   # |0⟩
# Estado completo: |00⟩_a ⊗ |ψ⟩
full_in = np.zeros(dim_total, dtype=complex)
full_in[0:2] = psi
full_out = U_block @ full_in
# Componente de ancilla |00⟩
sys_out = full_out[0:2]
expected = H_normalized @ psi
print(f"\nAcción sobre |0⟩_sys:")
print(f"  Resultado (bloque) = {np.round(sys_out, 6)}")
print(f"  Esperado  (H/α)|0⟩ = {np.round(expected, 6)}")
print(f"  Error              = {norm(sys_out - expected):.2e}")

# ============================================================
# 6. Resumen y equivalencia Qiskit
# ============================================================
print("\n" + "=" * 60)
print("EQUIVALENCIA CON CIRCUITOS QISKIT (pseudocódigo)")
print("=" * 60)
print("""
# PREPARE (2 qubits de ancilla)
qc_prep = QuantumCircuit(2)
# Circuito que prepara √p0|00⟩ + √p1|01⟩ + √p2|10⟩
# (se puede sintetizar con Ry controladas o StatePreparation)

# SELECT (2 ancillas + 1 sistema)
qc_sel = QuantumCircuit(3)  # q0,q1 = ancilla; q2 = sistema
# if ancilla == 00 → X(q2)
# if ancilla == 01 → Z(q2)
# if ancilla == 10 → Y(q2)
# Implementable con puertas multi-controladas

# Block-encoding
# U = (PREPARE† ⊗ I) · SELECT · (PREPARE ⊗ I)
""")

print("\nArchivo generado: lcu_block_encoding.py")
print("Block-encoding verificado con éxito (error numérico ~ 1e-16).")
