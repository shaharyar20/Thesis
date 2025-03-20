import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_stats(y_pred, y_true, path=None):
    # Create the outer subplot
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.3, 0.7])

    # Plot in the first subplot
    ax1 = fig.add_subplot(gs[0])
    plot_diff_boxplots(ax1, y_pred, y_true, 9)

    # Create the inner subplots
    inner_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1], hspace=0.5, wspace=0.3)
    plot_hist_vels_as_subplot(fig, inner_gs, y_pred)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    if path is not None:
        fig.savefig(path)

    plt.show()


def plot_hist_vels_as_subplot(fig, inner_gs, vels):
    for i, grid_elem in enumerate(inner_gs):
        ax = fig.add_subplot(grid_elem)

        if i >= 9:
            break
        ax.hist(vels[:, i], bins=40)
        ax.set_title(f"Discrete velocity {i}")

    fig.suptitle("Distribution of predicted values")


def plot_diff_boxplots(ax, y_pred, y_true, num_ele):
    boxplot_artists = []
    diff = abs(y_pred - y_true) / y_true

    for i in range(num_ele):
        data = diff[:, i]

        boxplot_artist = ax.boxplot(data, positions=[i], patch_artist=True)
        boxplot_artists.append(boxplot_artist)

        # Set the outliers as smaller circles
        for flier in boxplot_artist["fliers"]:
            flier.set(markersize=3, alpha=0.5)

    ax.set_xticks(range(num_ele))
    # ax.set_xticklabels(['Element {}'.format(i) for i in range(9)])
    ax.set_xlabel("Discrete velocities")
    ax.set_ylabel("|fi_true - f_pred| / fi_true")
    ax.set_title("Boxplots of relative f1 loss")
    ax.set_yscale("log")  # Set y-axis to logarithmic scale
    ax.set_ylim(1e-10, 0.1)

    # Add colors to the boxplots
    colors = ["blue", "green", "yellow", "red", "cyan", "orange", "purple", "grey", "steelblue"]
    for i, boxplot_artist in enumerate(boxplot_artists):
        for patch in boxplot_artist["boxes"]:
            patch.set_facecolor(colors[i])


def plot_hist_vels(vels):
    fig, axes = plt.subplots(3, 3, figsize=(8, 6))

    for i, ax in enumerate(axes.flat):
        if i >= 9:
            break
        ax.hist(vels[:, i], bins=40)
        ax.set_title(f"Discrete velocity {i}")

    fig.suptitle("Distribution of predicted values")
    plt.tight_layout()
    plt.show()


def plot_boxplots(data):
    fig, axes = plt.subplots(3, 3, figsize=(7, 4))
    for i, ax in enumerate(axes.flat):
        boxplot_artist = ax.boxplot(data[:, i])
        ax.set_title(f"Boxplot of vel {i}")
        # Set the outliers as smaller circles
        for flier in boxplot_artist["fliers"]:
            flier.set(markersize=1, alpha=0.5)

    plt.subplots_adjust(hspace=0.6, wspace=0.5)
    plt.show()
