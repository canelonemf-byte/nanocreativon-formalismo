from pathlib import Path
#!/usr/bin/env python3
"""
Formalismo Matemático y Protocolo de Simulación de Nanocreativones
LECN-LLP — Lázaro López Paz
El Prat de Llobregat, Barcelona, España
26-28 de julio de 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from numpy.linalg import norm
import json
from datetime import datetime, timezone

@dataclass
class NanoCreativonState:
    """Estado completo de un nanocreativón (híbrido cuántico-molecular-físico)."""
    psi: np.ndarray = field(default_factory=lambda: np.array([0.95, 0.05, 0.0, 0.0], dtype=np.complex128))
    style_seed: np.ndarray = field(default_factory=lambda: np.random.randn(32))
    cooccurrence: np.ndarray = field(default_factory=lambda: np.random.rand(16, 16) * 0.3)
    num_synapses: int = 100000
    folding_energy: float = 0.0
    optical_response: np.ndarray = field(default_factory=lambda: np.zeros(8))
    n_eff: float = 1.5
    absorption_peak: float = 1.0
    non_reciprocity: float = 0.0
    fitness: float = 0.0
    phi: float = 0.0
    generation_born: int = 0

    def __post_init__(self):
        self.psi = self.psi.astype(np.complex128)
        self._normalize_psi()
        self.style_seed = self.style_seed / (norm(self.style_seed) + 1e-12)

    def _normalize_psi(self):
        n = norm(self.psi)
        if n > 1e-12:
            self.psi /= n

    def probabilities(self) -> np.ndarray:
        return np.abs(self.psi)**2

    def fidelity_to_N4(self) -> float:
        target = np.array([0, 0, 0, 1], dtype=np.complex128)
        return float(np.abs(np.vdot(target, self.psi))**2)


def quantum_mutation_operator(mu: float, bias_high: float = 0.0) -> np.ndarray:
    """Operador unitario 4x4 para mutación del estado creativo."""
    theta = mu * np.pi * (0.3 + 0.7 * np.random.rand())
    phi = mu * np.pi * (0.3 + 0.7 * np.random.rand())

    def rx4(th):
        c, s = np.cos(th/2), np.sin(th/2)
        U = np.eye(4, dtype=np.complex128)
        U[0,0] = c; U[0,1] = -1j*s
        U[1,0] = -1j*s; U[1,1] = c
        return U

    def ry4(th):
        c, s = np.cos(th/2), np.sin(th/2)
        U = np.eye(4, dtype=np.complex128)
        U[0,0] = c; U[0,1] = -s
        U[1,0] = s; U[1,1] = c
        return U

    U = ry4(phi) @ rx4(theta)

    if bias_high > 0.05 and np.random.rand() < 0.4:
        angle = 0.12 * np.pi * bias_high
        mix = np.eye(4, dtype=np.complex128)
        c, s = np.cos(angle), np.sin(angle)
        mix[2,2] = c; mix[2,3] = -s
        mix[3,2] = s; mix[3,3] = c
        t = 0.06 * bias_high
        mix[0,0] = np.sqrt(max(0, 1-t)); mix[0,3] = np.sqrt(t)
        U = mix @ U

    if np.random.rand() < mu * 1.1:
        H = (1/np.sqrt(2)) * np.array([
            [1, 1, 0, 0], [1, -1, 0, 0],
            [0, 0, 1, 1], [0, 0, 1, -1]
        ], dtype=np.complex128)
        U = H @ U
    return U


def plasmid_crossover(p1: NanoCreativonState, p2: NanoCreativonState) -> NanoCreativonState:
    """Cruce por intercambio de plásmidos + entrelazamiento aproximado."""
    child = NanoCreativonState()
    alpha = 0.4 + 0.2 * np.random.rand()
    child.psi = alpha * p1.psi + (1-alpha) * p2.psi
    child.psi *= np.exp(1j * np.pi * np.random.rand())
    child._normalize_psi()

    cut = np.random.randint(8, 24)
    child.style_seed = np.concatenate([p1.style_seed[:cut], p2.style_seed[cut:]])
    child.style_seed /= (norm(child.style_seed) + 1e-12)

    beta = np.random.rand()
    child.cooccurrence = beta * p1.cooccurrence + (1-beta) * p2.cooccurrence
    child.cooccurrence += np.random.normal(0, 0.01, size=child.cooccurrence.shape)
    child.cooccurrence = np.clip(child.cooccurrence, 0, 1)
    child.num_synapses = int(0.5 * (p1.num_synapses + p2.num_synapses))
    return child


def evaluate_molecular_fitness(nano: NanoCreativonState) -> float:
    """Fitness: nivel creativo + entropía + plegado + metamateriales."""
    probs = nano.probabilities()
    level_score = 10*probs[0] + 30*probs[1] + 60*probs[2] + 95*probs[3]

    p = nano.cooccurrence.flatten()
    p = p / (p.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    entropy_score = 12 * (entropy / np.log(len(p)))

    folding_score = 8 * np.exp(-0.5 * ((nano.folding_energy + 8)**2) / 25)
    optical_score = 6 * min(1.0, norm(nano.optical_response) / 2.0)

    meta_score = 0.0
    if probs[3] > 0.3:
        meta_score += 10 * max(0, -nano.n_eff)
        meta_score += 5 * nano.non_reciprocity

    F = level_score + entropy_score + folding_score + optical_score + meta_score + np.random.normal(0, 2.0)
    F = float(np.clip(F, 0.0, 100.0))
    nano.fitness = F
    nano.phi = np.log10(1.0 + F)
    return F


def update_physical_properties(nano: NanoCreativonState):
    """Actualiza propiedades físicas emergentes (modelo fenomenológico)."""
    probs = nano.probabilities()
    nano.n_eff = 1.45 - 2.8 * probs[3] - 0.9 * probs[2] + 0.15 * np.random.randn()
    nano.absorption_peak = 0.8 + 4.2 * probs[3] + 1.5 * probs[2]
    nano.non_reciprocity = np.clip(0.1 + 0.85 * (probs[2] + probs[3]) + 0.05*np.random.randn(), 0, 1)
    nano.folding_energy = -12.0 + 6.0 * (1 - probs[3]) + np.random.normal(0, 1.5)
    base = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 0.6, 0.4, 0.25])
    nano.optical_response = base * (0.5 + 1.5 * probs[3]) + np.random.normal(0, 0.08, 8)
    nano.optical_response = np.clip(nano.optical_response, 0, 2)


class NanoCreativonSwarm:
    """Enjambre evolutivo de nanocreativones."""

    def __init__(self, pop_size: int = 48, mu0: float = 0.2):
        self.pop_size = pop_size
        self.mu0 = mu0
        self.population: List[NanoCreativonState] = []
        self.generation = 0
        self.history: List[Dict] = []
        self.best_ever: Optional[NanoCreativonState] = None
        self.best_phi = -1.0

        for _ in range(pop_size):
            nano = NanoCreativonState()
            nano.psi = np.array([0.92, 0.06, 0.015, 0.005], dtype=np.complex128)
            nano._normalize_psi()
            update_physical_properties(nano)
            self.population.append(nano)

    def mutation_rate(self, g: int) -> float:
        return self.mu0 / np.log10(g + 10)

    def step(self) -> Dict:
        g = self.generation
        mu = self.mutation_rate(g)

        fitnesses, phis, fids = [], [], []
        for nano in self.population:
            update_physical_properties(nano)
            F = evaluate_molecular_fitness(nano)
            fitnesses.append(F)
            phis.append(nano.phi)
            fids.append(nano.fidelity_to_N4())

        fitnesses = np.array(fitnesses)
        phis = np.array(phis)
        fids = np.array(fids)

        n_elite = max(4, self.pop_size // 8)
        elite_idx = np.argsort(phis)[-n_elite:][::-1]
        elites = []
        for i in elite_idx:
            e = NanoCreativonState(
                psi=self.population[i].psi.copy(),
                style_seed=self.population[i].style_seed.copy(),
                cooccurrence=self.population[i].cooccurrence.copy(),
                num_synapses=self.population[i].num_synapses,
                folding_energy=self.population[i].folding_energy,
                optical_response=self.population[i].optical_response.copy(),
                n_eff=self.population[i].n_eff,
                absorption_peak=self.population[i].absorption_peak,
                non_reciprocity=self.population[i].non_reciprocity,
                fitness=self.population[i].fitness,
                phi=self.population[i].phi,
                generation_born=self.population[i].generation_born
            )
            elites.append(e)

        if phis[elite_idx[0]] > self.best_phi:
            self.best_phi = phis[elite_idx[0]]
            self.best_ever = elites[0]

        amplified = np.exp(2.5 * (phis - phis.min()))
        probs_sel = amplified / (amplified.sum() + 1e-12)
        parent_indices = np.random.choice(len(phis), size=self.pop_size, p=probs_sel)

        new_pop = elites[:]
        while len(new_pop) < self.pop_size:
            i1, i2 = np.random.choice(parent_indices, size=2, replace=False)
            child = plasmid_crossover(self.population[i1], self.population[i2])
            bias = min(1.0, 0.12 + 0.002 * g + 0.2 * child.phi)
            U = quantum_mutation_operator(mu, bias_high=bias)
            child.psi = U @ child.psi
            child._normalize_psi()
            if np.random.rand() < mu:
                child.style_seed += np.random.normal(0, 0.03*mu, size=32)
                child.style_seed /= (norm(child.style_seed) + 1e-12)
            if np.random.rand() < mu:
                child.cooccurrence += np.random.normal(0, 0.012*mu, size=child.cooccurrence.shape)
                child.cooccurrence = np.clip(child.cooccurrence, 0, 1)
            child.generation_born = g
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        self.generation += 1

        stats = {
            "generation": g, "mu": float(mu),
            "mean_F": float(fitnesses.mean()), "max_F": float(fitnesses.max()),
            "mean_phi": float(phis.mean()), "max_phi": float(phis.max()),
            "mean_fid_N4": float(fids.mean()), "max_fid_N4": float(fids.max()),
            "best_n_eff": float(self.best_ever.n_eff) if self.best_ever else 0.0
        }
        self.history.append(stats)
        return stats

    def run(self, max_gen: int = 180, target_fid: float = 0.50) -> Dict:
        print(f"=== Enjambre de Nanocreativones | N={self.pop_size} | MaxGen={max_gen} ===")
        for g in range(max_gen):
            stats = self.step()
            if g % 20 == 0 or g == max_gen-1:
                print(f"Gen {stats['generation']:3d} | μ={stats['mu']:.4f} | "
                      f"Fmax={stats['max_F']:.1f} | FidN4={stats['mean_fid_N4']:.3f}/{stats['max_fid_N4']:.3f} | "
                      f"n_eff(best)={stats['best_n_eff']:.3f}")
            if stats['mean_fid_N4'] > target_fid and g > 40:
                recent = [h['mean_fid_N4'] for h in self.history[-5:]]
                if all(f > target_fid * 0.92 for f in recent):
                    print(f"*** Criterio de fidelidad alcanzado en generación {g} ***")
                    break
        return self.summary()

    def summary(self) -> Dict:
        if not self.history:
            return {}
        last = self.history[-1]
        best = self.best_ever
        return {
            "final_generation": last["generation"],
            "final_mean_fid_N4": last["mean_fid_N4"],
            "final_max_fid_N4": last["max_fid_N4"],
            "final_max_phi": last["max_phi"],
            "best_n_eff": float(best.n_eff) if best else None,
            "best_absorption_peak": float(best.absorption_peak) if best else None,
            "best_non_reciprocity": float(best.non_reciprocity) if best else None,
            "best_level_probs": best.probabilities().tolist() if best else None,
            "best_folding_energy": float(best.folding_energy) if best else None
        }


if __name__ == "__main__":
    np.random.seed(42)
    swarm = NanoCreativonSwarm(pop_size=48, mu0=0.2)
    summary = swarm.run(max_gen=180, target_fid=0.50)

    print("\n" + "="*60)
    print("RESUMEN FINAL DEL ENJAMBRE DE NANOCREATIVONES")
    print("="*60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    report = {
        "titulo": "Simulación de Enjambre de Nanocreativones — LECN-LLP",
        "fecha": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "autor": "Lázaro López Paz",
        "origen": "El Prat de Llobregat, Barcelona, España",
        "resultados": summary,
        "notas": [
            "Formalismo híbrido: estado cuántico de niveles + parámetros moleculares + propiedades físicas emergentes.",
            "Cruce implementado como intercambio de plásmidos + entrelazamiento aproximado.",
            "Mutación mediante operadores unitarios modulados por tasa logarítmica μ(g).",
            "Fitness combina nivel creativo, entropía estructural, plegado y propiedades metamateriales.",
            "El índice de refracción efectivo tiende a valores negativos en alta fidelidad a N4."
        ]
    }
    # ensure output dir
    Path(__file__).resolve().parent.joinpath("artifacts").mkdir(parents=True, exist_ok=True)
    (Path(__file__).resolve().parent / "artifacts" / "LECN_LLP_Unificado" / "nanocreativones").mkdir(parents=True, exist_ok=True)
    with open(str(Path(__file__).resolve().parent / "artifacts" / "LECN_LLP_Unificado" / "nanocreativones" / "nanocreativon_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nReporte guardado en nanocreativon_report.json")
