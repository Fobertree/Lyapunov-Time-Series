import matplotlib.pyplot as plt
import numpy as np
import os

# def plot_data(data : np.array, label: str, title:str, color: str, save_path : str = None, impute_filter : np.array = None):
#     x = np.arange(len(data)) # to do: replace with date, although returning fig object we can do manip later
#     fig, ax = plt.subplots()

#     ax.plot(x[~impute_filter], data[~impute_filter], label=label, color=color)
    
#     if isinstance(impute_filter, np.ndarray):
#         # better is to split each segment into contiguous non-nan impute, then for loop plot
#         ax.plot(x[impute_filter], data[impute_filter], label="Imputed", c = "red", linestyle = "None", marker = 'o')
    
#     ax.set_xlabel("Time Index")
#     ax.set_ylabel("Lyapunov Exponent")
#     ax.set_title(title)
#     ax.grid(True)
#     ax.legend()
#     plt.tight_layout()
    
    
#     if save_path:
#         fig.savefig(os.path.join("Plots", save_path))
#     else:
#         fig.show()

#     return fig

def plot_data(
    data: np.array, 
    label: str, 
    title: str, 
    color: str, 
    save_path: str = None, 
    impute_filter: np.array = None,
    orig_series: np.array = None  # new arg
):
    x = np.arange(len(data))  # later replace with dates if needed
    fig, ax = plt.subplots()

    # Plot Lyapunov Exponent Time Series
    ax.plot(x[~impute_filter], data[~impute_filter], label=label, color=color)

    if isinstance(impute_filter, np.ndarray):
        ax.plot(x[impute_filter], data[impute_filter], label="Imputed", 
                c="red", linestyle="None", marker='o')

    ax.set_xlabel("Time Index")
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title(title)
    ax.grid(True)

    # Optional: plot original time series on a secondary y-axis
    if orig_series is not None:
        ax2 = ax.twinx()
        ax2.plot(x, orig_series[-len(x):], label="Original Series", color='gray', 
                 linestyle='--', alpha=0.5)
        ax2.set_ylabel("Original Time Series", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    if orig_series is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax.legend(lines, labels)

    plt.tight_layout()

    if save_path:
        fig.savefig(os.path.join("Plots", save_path))
    else:
        fig.show()

    return fig