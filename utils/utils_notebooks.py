import numpy as np
import matplotlib.pyplot as plt


def display_ps(classes: list, labels: list) -> str:
    if not labels:
        return "{}"
    else:
        str_ps = "{" + classes[labels[0]]

    if len(labels) > 1:
        for l in labels[1:]:
            str_ps += ", " + classes[l]

    str_ps += "}"
    return str_ps


def to_img(x):
    return (x - x.min()) / (x.max() - x.min())


def amplitude(x):
    x = np.sum(x**2, axis=-1)
    x = (x - x.min()) / (x.max() - x.min())
    x = np.log(x + 1)
    x = np.expand_dims(x, axis=-1)
    return x


def plot_banner(inputs, vanilla_ps, robust_ps, classes):
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(to_img(inputs[i]).cpu().numpy().transpose(1, 2, 0))
        lb_ps = [k for k in range(10) if vanilla_ps[i, k]]
        lb_ps_rob = [k for k in range(10) if robust_ps[i, k]]
        plt.title(
            f"PS : {display_ps(classes, lb_ps)}\nRPS : {display_ps(classes, lb_ps_rob)}"
        )
        plt.axis("off")
    plt.show()


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def plot_vanilla_bounds_banner(EPSILONS, ALPHA, DELTA, gamma_mins, gamma_maxs):
    import seaborn as sns

    sns.set()

    plt.figure(figsize=(11, 6))
    plt.plot(EPSILONS, gamma_mins, "ro--", label="lower bound")
    plt.plot(EPSILONS, gamma_maxs, "bo--", label="upper bound")
    plt.plot(
        EPSILONS,
        [1 - ALPHA for _ in EPSILONS],
        label="Desired coverage",
        linestyle="--",
        color="black",
    )
    plt.fill_between(EPSILONS, gamma_mins, gamma_maxs, color="gray", alpha=0.5)

    plt.xlabel("Epsilon")
    plt.ylabel("Probability")
    plt.title(
        f"Vanilla CP coverage bounds for different $\epsilon$ values with probability {(1 - DELTA) * 100}%"
    )
    plt.legend()
