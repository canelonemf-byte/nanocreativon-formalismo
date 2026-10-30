#!/usr/bin/env python3
"""
Cálculo numérico del número de enrollamiento (winding number)
del modelo de Hatano-Nelson.

LECN-LLP — Lázaro López Paz
El Prat de Llobregat, Barcelona, España
28 de julio de 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def E_hatano_nelson(k, t_R, t_L):
    """Espectro de Bloch del modelo de Hatano-Nelson."""
    return t_R * np.exp(-1j * k) + t_L * np.exp(1j * k)


def winding_number(t_R, t_L, n_points=4000, E0=0.0, tol=1e-10):
    """
    Calcula el winding number numéricamente.

    W = (1 / 2π) * Δ arg(E(k) - E0)  mientras k recorre [0, 2π).

    Para el caso hermítico (t_R ≈ t_L) el espectro es real y W = 0.
    """
    # Caso hermítico (o casi): espectro real → no hay point gap alrededor de 0
    if abs(t_R - t_L) < tol:
        return 0, 0.0

    k = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    E = E_hatano_nelson(k, t_R, t_L) - E0

    # Evitar el origen exacto (puede producir saltos numéricos)
    mask = np.abs(E) > 1e-14
    if not np.any(mask):
        return 0, 0.0

    phase = np.unwrap(np.angle(E))

    # Variación total de fase a lo largo de la zona de Brillouin
    # Como el camino es cerrado, comparamos el final con el inicio
    total_phase_change = phase[-1] - phase[0]

    # Corrección de cierre (el punto k=2π debe coincidir con k=0)
    # Añadimos el salto residual si existe
    E_first = E[0]
    E_last_close = E_hatano_nelson(2*np.pi, t_R, t_L) - E0  # debería = E[0]
    # Con endpoint=False ya tenemos buena cobertura; unwrap da la variación continua

    W = total_phase_change / (2 * np.pi)
    return int(np.round(W)), float(total_phase_change)


def plot_spectrum(t_R, t_L, ax=None, label=None):
    """Dibuja el espectro E(k) en el plano complejo."""
    k = np.linspace(0, 2*np.pi, 600)
    E = E_hatano_nelson(k, t_R, t_L)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(E.real, E.imag, lw=2, label=label)
    ax.plot(0, 0, 'kx', markersize=9, markeredgewidth=2)
    ax.set_xlabel('Re(E)')
    ax.set_ylabel('Im(E)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4)
    return ax


if __name__ == "__main__":
    print("=" * 62)
    print("Número de enrollamiento — Modelo de Hatano-Nelson")
    print("LECN-LLP · Lázaro López Paz · El Prat de Llobregat")
    print("=" * 62)

    cases = [
        (1.0, 1.0, "Hermítico (t_R = t_L)"),
        (1.5, 0.5, "t_R > t_L"),
        (0.5, 1.5, "t_R < t_L"),
        (2.0, 0.0, "Solo hopping a la derecha"),
        (0.0, 2.0, "Solo hopping a la izquierda"),
        (1.2, 0.8, "Asimetría moderada"),
    ]

    print(f"\n{'t_R':>6} {'t_L':>6}  {'W':>4}  {'Δphase/2π':>10}  Descripción")
    print("-" * 62)

    results = []
    for t_R, t_L, desc in cases:
        W, phase_change = winding_number(t_R, t_L)
        print(f"{t_R:6.2f} {t_L:6.2f}  {W:4d}  {phase_change/(2*np.pi):10.4f}  {desc}")
        results.append((t_R, t_L, W, desc))

    # Visualización
    out_dir = (Path(__file__).resolve().parent / "artifacts" / "LECN_LLP_Unificado" / "nanocreativones")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (t_R, t_L, W, desc) in zip(axes, results):
        plot_spectrum(t_R, t_L, ax=ax)
        ax.set_title(f"{desc}\nW = {W}", fontsize=10)
        if abs(t_R - t_L) > 1e-10:
            ax.legend(['E(k)', 'E0 = 0'], fontsize=8, loc='best')
        else:
            ax.legend(['E(k) (real)', 'E0 = 0'], fontsize=8, loc='best')

    fig.suptitle("Espectro complejo del modelo de Hatano-Nelson\n"
                 "Número de enrollamiento W (point gap alrededor de E = 0)",
                 fontsize=13, fontweight='bold', color='#1a365d')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / "hatano_nelson_winding.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nFigura guardada: {out_path}")

    print("\n" + "=" * 62)
    print("Ejemplo de uso de la función:")
    print("=" * 62)
    print("""
from winding_number_hatano_nelson import winding_number

W, phase = winding_number(t_R=1.5, t_L=0.5)
print(W)          # → -1
""")
    W, _ = winding_number(1.5, 0.5)
    print(f"winding_number(t_R=1.5, t_L=0.5) → W = {W}")
    print("W ≠ 0  ⇒  existe point gap  ⇒  efecto piel no hermítico bajo OBC.")
