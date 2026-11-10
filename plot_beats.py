import matplotlib.pyplot as plt
CLASS_NAMES = {0:"N - Normal",1:"S - Supraventricular",2:"V - Ventricular",3:"F - Fusion",4:"Q - Unknown/Paced"}
def plot_examples(X, y, out_path):
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for c, ax in enumerate(axes):
        i = next(i for i, label in enumerate(y) if int(label) == c)
        ax.plot(X[i], linewidth=1.2); ax.set_title(CLASS_NAMES[c]); ax.set_ylabel("Amplitude"); ax.grid(alpha=.25)
    axes[-1].set_xlabel("Sample index (125 Hz)")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
