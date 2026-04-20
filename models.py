"""Jednoduché ukázky numerických modelů.

V souboru najdete tři úlohy:
1. Základní epidemiologický model SIR.
2. Rozšířený model dravec--kořist s konkurenčním druhem.
3. Monte Carlo simulace pro odhad čísla pí.

Vše je psáno co nejjednodušeji a využívá knihovny ``numpy`` a
``matplotlib`` pro výpočty a vykreslení grafů.
"""

import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# SIR MODEL
# ----------------------------------------------------------------------

def simulate_sir(
    beta: float,
    gamma: float,
    s0: float,
    i0: float,
    r0: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simuluje šíření nákazy jednoduchým Eulerovým krokem."""
    S = [s0]
    I = [i0]
    R = [r0]
    N = s0 + i0 + r0
    for _ in range(steps):
        s, i, r = S[-1], I[-1], R[-1]
        ds = -beta * s * i / N
        di = beta * s * i / N - gamma * i
        dr = gamma * i
        S.append(s + ds * dt)
        I.append(i + di * dt)
        R.append(r + dr * dt)
    t = np.linspace(0, dt * steps, steps + 1)
    return t, np.vstack([S, I, R])


def plot_sir(t: np.ndarray, data: np.ndarray) -> None:
    """Vykreslí průběh modelu SIR."""
    plt.figure()
    plt.plot(t, data[0], label="S")
    plt.plot(t, data[1], label="I")
    plt.plot(t, data[2], label="R")
    plt.xlabel("čas")
    plt.ylabel("počet osob")
    plt.title("Model SIR")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# LOTKA-VOLTERRA S KONKURENTEM
# ----------------------------------------------------------------------

def simulate_lv(
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    prey0: float,
    pred0: float,
    comp_rate: float,
    comp0: float,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rozšířený model dravec--kořist."""
    prey = [prey0]
    pred = [pred0]
    comp = [comp0]
    for _ in range(steps):
        x, y, c = prey[-1], pred[-1], comp[-1]
        dx = alpha * x - beta * x * (y + c)
        dy = delta * x * y - gamma * y
        dc = comp_rate * x * c - gamma * c
        prey.append(x + dx * dt)
        pred.append(y + dy * dt)
        comp.append(c + dc * dt)
    t = np.linspace(0, dt * steps, steps + 1)
    return t, np.vstack([prey, pred, comp])


def plot_lv(t: np.ndarray, data: np.ndarray) -> None:
    """Vykreslí průběh rozšířeného Lotka--Volterra modelu."""
    plt.figure()
    plt.plot(t, data[0], label="kořist")
    plt.plot(t, data[1], label="dravci")
    plt.plot(t, data[2], label="konkurent")
    plt.xlabel("čas")
    plt.ylabel("počet")
    plt.title("Lotka–Volterra s konkurencí")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# MONTE CARLO OD HAD PI
# ----------------------------------------------------------------------

def monte_carlo_pi(samples: int = 10000) -> float:
    """Odhadne číslo pí pomocí náhodného vzorkování."""
    inside_x = []
    inside_y = []
    outside_x = []
    outside_y = []
    inside = 0
    for _ in range(samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)
    pi_est = 4 * inside / samples

    plt.figure()
    plt.scatter(inside_x, inside_y, s=4, color="tab:blue")
    plt.scatter(outside_x, outside_y, s=4, color="tab:red")
    plt.gca().set_aspect("equal")
    plt.title(f"Monte Carlo odhad π ≈ {pi_est:.4f}")
    plt.tight_layout()
    plt.show()
    return pi_est


# ----------------------------------------------------------------------
# MAIN - spustí ukázkové simulace
# ----------------------------------------------------------------------

def main() -> None:
    t, data = simulate_sir(beta=0.3, gamma=0.1, s0=990, i0=10, r0=0, dt=0.1, steps=160)
    plot_sir(t, data)

    t, data = simulate_lv(
        alpha=1.0,
        beta=0.1,
        delta=0.075,
        gamma=1.5,
        prey0=40,
        pred0=9,
        comp_rate=0.05,
        comp0=5,
        dt=0.1,
        steps=500,
    )
    plot_lv(t, data)

    pi_est = monte_carlo_pi(5000)
    print(f"Odhadnuté π: {pi_est:.4f}")


if __name__ == "__main__":
    main()
