# src/plot_results.py
import matplotlib.pyplot as plt
from IPython.display import display  # <-- added

def plot_fdi_attack_detection(
    action_probs_test_phi35,
    action_probs_test_theta46,
    action_probs_test_act_Theta55,
    action_probs_test_act_Psi66,
    save_path="FDIAttackDetection.pdf"
):
    fig, axs = plt.subplots(2, 1, figsize=(7, 7))

    axs[0].plot(action_probs_test_phi35, label=r'$\phi$')
    axs[0].plot(action_probs_test_theta46, label=r'$\Theta$')
    axs[0].axhline(y=0.95, color='r', linestyle='--', label='Detection Threshold')
    axs[0].set_ylabel('Sensor Attack Probability', fontsize=16)
    axs[0].set_title('FDI Attack Detection and Isolation', fontsize=18)
    axs[0].legend(fontsize=14)

    axs[1].plot(action_probs_test_act_Theta55, label=r'$\bar{\Theta}$')
    axs[1].plot(action_probs_test_act_Psi66, label=r'$\bar{\psi}$')
    axs[1].axhline(y=0.95, color='r', linestyle='--')
    axs[1].set_xlabel('Time (s)', fontsize=14)
    axs[1].set_ylabel('Actuator Attack Probability', fontsize=16)
    axs[1].legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path)
    display(fig)  # <-- ensure shown in Jupyter
    plt.close(fig)  # <-- close to avoid duplicate display in notebooks


def plot_covert_attack_detection(
    action_probs_test_act_Theta25_Covert,
    action_probs_test_act_Psi45_Covert,
    save_path="CovertAttackDetection.pdf"
):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(action_probs_test_act_Theta25_Covert, label=r'$\bar{\psi}$')
    plt.plot(action_probs_test_act_Psi45_Covert, label=r'$\bar{\Theta}$')
    plt.axhline(y=0.95, color='r', linestyle='--', label='Detection Threshold')
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Cyberattack Probability', fontsize=16)
    plt.title('Covert Attack Detection and Isolation', fontsize=16)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    display(fig)  # <-- ensure shown in Jupyter
    plt.close(fig)


def plot_performance_comparison(
    action_probs_test_theta43,
    detection_vector_theta43RL,
    df_chi43,
    df_Euc43,
    action_probs_test_theta43_sysnoise,
    detection_vector_theta43_sysnoiseRL,
    df_chi43sysnoise,
    df_Euc43sysnoise,
    action_probs_test_theta43_attacknoise,
    detection_vector_theta43_attacknoiseRL,
    df_chi43attacknoise,
    df_Euc43attacknoise,
    save_path="DetectionComparison.pdf"
):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10.2))

    axs[0].plot(action_probs_test_theta43, label='DRL (Probability)')
    axs[0].plot(detection_vector_theta43RL, label='RL')
    axs[0].plot(df_chi43['time'], df_chi43['thr-lev'], linestyle="-", label=r'$\chi^2$')
    axs[0].plot(df_Euc43['time'], df_Euc43['thr-lev'], linestyle="-", label="Euclidean")
    axs[0].axhline(y=0.95, color='r', linestyle='--', label='Detection Threshold')
    axs[0].set_ylabel('Normal Noise', fontsize=14.5)
    axs[0].set_title('Performance Comparison ', fontsize=16)
    axs[0].legend(fontsize=14)

    axs[1].plot(action_probs_test_theta43_sysnoise, label='DRL')
    axs[1].plot(detection_vector_theta43_sysnoiseRL, label='RL')
    axs[1].plot(df_chi43sysnoise['time'], df_chi43sysnoise['thr-lev'], linestyle="-", label="Chi")
    axs[1].plot(df_Euc43sysnoise['time'], df_Euc43sysnoise['thr-lev'], linestyle="-", label="Euc")
    axs[1].axhline(y=0.95, color='r', linestyle='--', label='Detection Threshold')
    axs[1].set_ylabel('Increased Measurement Noise', fontsize=14.5)

    axs[2].plot(action_probs_test_theta43_attacknoise, label='DRL')
    axs[2].plot(detection_vector_theta43_attacknoiseRL, label='RL')
    axs[2].plot(df_chi43attacknoise['time'], df_chi43attacknoise['thr-lev'], linestyle="-", label="chi")
    axs[2].plot(df_Euc43attacknoise['time'], df_Euc43attacknoise['thr-lev'], linestyle="-", label="Euclidean")
    axs[2].axhline(y=0.95, color='r', linestyle='--', label='Detection Threshold')
    axs[2].set_xlabel('Time (s)', fontsize=16)
    axs[2].set_ylabel('Increased Attack Noise', fontsize=14.5)

    plt.tight_layout()
    plt.savefig(save_path)
    display(fig)  # <-- ensure shown in Jupyter
    plt.close(fig)
