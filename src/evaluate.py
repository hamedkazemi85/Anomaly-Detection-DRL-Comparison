import matplotlib.pyplot as plt
import numpy as np
import os

def plot_fdi_detection(prob_phi, prob_theta, prob_act_theta55, prob_act_psi66, out_path="FDIAttackDetection.pdf"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 7))
    axs[0].plot(prob_phi, label=r'$\phi$')
    axs[0].plot(prob_theta, label=r'$\Theta$')
    axs[0].axhline(y=0.95, color='r', linestyle='--', label='Detection Threshold')
    axs[0].set_ylabel('Sensor Attack Probability', fontsize=12)
    axs[0].set_title('FDI Attack Detection and Isolation', fontsize=14)
    axs[0].legend()

    axs[1].plot(prob_act_theta55, label=r'$\bar{\Theta}$')
    axs[1].plot(prob_act_psi66, label=r'$\bar{\psi}$')
    axs[1].axhline(y=0.95, color='r', linestyle='--')
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Actuator Attack Probability', fontsize=12)
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_comparison_block(prob_drl, detect_rl_vec, df_chi, df_euc, out_path="DetectionComparison.pdf", title="Performance Comparison"):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10.2))
    # first subplot: normal noise
    axs[0].plot(prob_drl, label='DRL (Probability)')
    axs[0].plot(detect_rl_vec, label='RL')
    axs[0].plot(df_chi['time'], df_chi['thr-lev'], linestyle="-", label=r'$\chi^2$')
    axs[0].plot(df_euc['time'], df_euc['thr-lev'], linestyle="-", label="Euclidean")
    axs[0].axhline(y=0.95, color='r', linestyle='--', label='Detection Threshold')
    axs[0].set_ylabel('Normal Noise', fontsize=12.5)
    axs[0].set_title(title, fontsize=14)
    axs[0].legend(fontsize=10)

    # second subplot and third subplot will be set by caller similarly if needed
    # For compactness, caller can re-use this function and pass preselected arrays to plot.
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
