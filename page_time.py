# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100  # Number of modes
T = 10.0  # Evaporation timescale
sigma = 5.0  # Width for Gaussian
C_alpha = 1.0  # Normalization constant
epsilon = 1e-12  # Cutoff for log
t = np.linspace(0.1, T, 300)  # Time grid
s = np.arange(1, N + 1)  # Mode indices
s1, s2 = 10, 90  # Firewall model parameters
lambda_res = 15.0  # Residue decay parameter

# Model A: Gaussian Sweep (Unitary)
mu = (N / 2) * (1 - np.cos(np.pi * t / T))
ws_A = np.array([np.exp(-(s[:, None] - mu)**2 / (2 * sigma**2)) for s in [s]]).squeeze()
Theta_A = np.sum(np.exp(-t * C_alpha * ws_A**2), axis=0)
S_sym_A = -np.log(np.maximum(Theta_A, epsilon))

# Model B: Residue (Information Loss)
ws_B = np.array([0.4 * np.exp(-s / lambda_res) + 0.6 * np.exp(-(s[:, None] - mu)**2 / (2 * sigma**2)) for s in [s]]).squeeze()
Theta_B = np.sum(np.exp(-t * C_alpha * ws_B**2), axis=0)
S_sym_B = -np.log(np.maximum(Theta_B, epsilon))

# Model C: Remnant (Entropy Freezing)
sigma_t = sigma * np.exp(-t / T)
ws_C = np.array([np.exp(-(s[:, None] - mu)**2 / (2 * sigma_t**2)) for s in [s]]).squeeze()
Theta_C = np.sum(np.exp(-t * C_alpha * ws_C**2), axis=0)
S_sym_C = -np.log(np.maximum(Theta_C, epsilon))

# Model D: Firewall (Discontinuous Transition)
ws_D = np.where(t < T/2, np.exp(-(s[:, None] - s1)**2 / (2 * sigma**2)), np.exp(-(s[:, None] - s2)**2 / (2 * sigma**2)))
Theta_D = np.sum(np.exp(-t * C_alpha * ws_D**2), axis=0)
S_sym_D = -np.log(np.maximum(Theta_D, epsilon))

# Find spectral Page times (approximate by max entropy)
t_page = [t[np.argmax(S_sym)] for S_sym in [S_sym_A, S_sym_B, S_sym_C, S_sym_D]]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(t, S_sym_A, label='Model A (Unitary)', color='#1f77b4')
plt.plot(t, S_sym_B, label='Model B (Residue)', color='#ff7f0e')
plt.plot(t, S_sym_C, label='Model C (Remnant)', color='#2ca02c')
plt.plot(t, S_sym_D, label='Model D (Firewall)', color='#d62728')
for tp in t_page:
    plt.axvline(tp, linestyle='--', color='black', alpha=0.5)
plt.xlabel('Time (t/T)')
plt.ylabel('Symbolic Entropy $S_{\\mathrm{sym}}(t)$')
plt.title('Symbolic Entropy Curves for Evaporation Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('symbolic_evaporation_panel.pdf', dpi=300, format='pdf')
plt.show()
