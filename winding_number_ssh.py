#!/usr/bin/env python3
"""
Cálculo numérico del número de enrollamiento (winding number / Zak phase)
del modelo SSH (Su-Schrieffer-Heeger).

LECN-LLP — Lázaro López Paz
El Prat de Llobregat, Barcelona, España
28 de julio de 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def d_vector(k, v, w):
    """Vector d(k) del modelo SSH: H(k) = d(k) · σ"""
    dx = v + w * np.cos(k)
    dy = w * np.sin(k)
    return dx, dy


def E_ssh(k, v, w):
    """Espectro del modelo SSH."""
    dx, dy = d_vector(k, v, w)
    return np.sqrt(dx**2 + dy**2)


def winding_number_ssh(v, w, n_points=4000, tol=1e-12):
    """
    Calcula el winding number del modelo SSH.

    W = (1 / 2π) ∫ ∂k φ(k) dk
    donde φ(k) = arg(dx + i dy) = arg(v + w e^{-ik})

    Retorna
    -------
    W : int
        0 (fase trivial) o 1 (fase topológica)  [en valor absoluto]
    phase_change : float
        Variación total de fase en radianes.
    """
    # Punto crítico: gap cerrado
    if abs(abs(v) - abs(w)) < tol:
        return 0, 0.0   # gap closure, winding no bien definido de la misma forma

    k = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    dx, dy = d_vector(k, v, w)

    # Fase del vector d en el plano (dx, dy)
    phase = np.unwrap(np.arctan2(dy, dx))

    total_phase_change = phase[-1] - phase[0]
    W = total_phase_change / (2 * np.pi)

    # El winding del SSH es 0 o ±1; tomamos el valor redondeado
    return int(np.round(W)) % 2, float(total_phase_change)
    # (usamos % 2 porque la orientación puede dar -1, que es equivalente a 1 en Z_2)


def plot_d_vector(v, w, ax=None):
    """Trayectoria de d(k) en el plano dx-dy."""
    k = np.linspace(0, 2*np.pi, 500)
    dx, dy = d_vector(k, v, w)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(dx, dy, lw=2)
    ax.plot(0, 0, 'kx', markersize=9, markeredgewidth=2)
    # Marcar inicio
    ax.plot(dx[0], dy[0], 'o', color='C0', markersize=6)
    ax.set_xlabel(r'$d_x$')
    ax.set_ylabel(r'$d_y$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4)
    return ax


def plot_spectrum(v, w, ax=None):
    """Espectro de bandas E(k)."""
    k = np.linspace(0, 2*np.pi, 500)
    E = E_ssh(k, v, w)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(k, E, 'b', lw=1.5, label=r'$E_+$')
    ax.plot(k, -E, 'r', lw=1.5, label=r'$E_-$')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$E$')
    ax.set_xlim(0, 2*np.pi)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    return ax


if __name__ == "__main__":
    print("=" * 64)
    print("Número de enrollamiento — Modelo SSH (Su-Schrieffer-Heeger)")
    print("LECN-LLP · Lázaro López Paz · El Prat de Llobregat")
    print("=" * 64)

    cases = [
        (1.5, 0.5, "Trivial  (|v| > |w|)"),
        (0.5, 1.5, "Topológico (|v| < |w|)"),
        (1.0, 1.0, "Punto crítico (|v| = |w|)"),
        (0.8, 1.2, "Topológico moderado"),
        (2.0, 0.5, "Trivial fuerte"),
        (0.3, 1.0, "Topológico fuerte"),
    ]

    print(f"\n{'v':>6} {'w':>6}  {'W':>4}  {'Δphase/2π':>10}  Fase")
    print("-" * 64)

    results = []
    for v, w, desc in cases:
        W, phase_change = winding_number_ssh(v, w)
        # Interpretación
        if abs(abs(v) - abs(w)) < 1e-12:
            fase = "Crítico (gap cerrado)"
        elif abs(v) > abs(w):
            fase = "Trivial"
        else:
            fase = "Topológico"
        print(f"{v:6.2f} {w:6.2f}  {W:4d}  {phase_change/(2*np.pi):10.4f}  {desc}")
        results.append((v, w, W, desc, fase))

    # Visualización
    out_dir = (Path(__file__).resolve().parent / "artifacts" / "LECN_LLP_Unificado" / "nanocreativones")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (v, w, W, desc, fase) in zip(axes, results):
        plot_d_vector(v, w, ax=ax)
        title_color = '#c53030' if fase == "Topológico" else ('#718096' if "Crítico" in fase else '#2b6cb0')
        ax.set_title(f"{desc}\nW = {W}  ({fase})", fontsize=10, color=title_color)

    fig.suptitle("Modelo SSH — Trayectoria de d(k) en el plano (dₓ, dᵧ)\n"
                 "W = 1 → fase topológica (el origen está rodeado) · W = 0 → fase trivial",
                 fontsize=13, fontweight='bold', color='#1a365d')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / "ssh_winding.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nFigura guardada: {out_path}")

    # Figura adicional: espectro de bandas para casos representativos
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    plot_spectrum(1.5, 0.5, ax=axes2[0])
    axes2[0].set_title("Fase trivial (v=1.5, w=0.5)\nW = 0", color='#2b6cb0')
    plot_spectrum(0.5, 1.5, ax=axes2[1])
    axes2[1].set_title("Fase topológica (v=0.5, w=1.5)\nW = 1", color='#c53030')
    fig2.suptitle("Espectro de bandas del modelo SSH", fontsize=12, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    out_path2 = out_dir / "ssh_bands.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Figura de bandas guardada: {out_path2}")

    print("\n" + "=" * 64)
    print("Ejemplo de uso:")
    print("=" * 64)
    print("""
from winding_number_ssh import winding_number_ssh

W, phase = winding_number_ssh(v=0.5, w=1.5)
print(W)          # → 1  (fase topológica)
""")
    W, _ = winding_number_ssh(0.5, 1.5)
    print(f"winding_number_ssh(v=0.5, w=1.5) → W = {W}")
    print("W = 1  ⇒  fase topológica  ⇒  estados de borde protegidos en cadena finita.")
