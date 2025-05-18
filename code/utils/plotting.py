import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def multiModelComparison(model_metrics, model_names):
    # Assumes each model_metrics[i] is a dict with the same keys (metrics)
    metric_keys = list(model_metrics[0].keys())
    num_metrics = len(metric_keys)
    num_models = len(model_names)
    bar_width = 0.8 / num_metrics  # Keep bars within each group

    x = np.arange(num_models)  # base x positions for model names

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, metric in enumerate(metric_keys):
        metric_values = [model[metric] for model in model_metrics]
        bar_positions = x + i * bar_width
        ax.bar(bar_positions, metric_values, width=bar_width, label=metric)

    # Set axis labels and titles
    ax.set_xlabel("Models")
    ax.set_ylabel("Metric Values")
    ax.set_title("Model Comparison by Metrics")
    ax.set_xticks(x + bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis="y")

    plt.tight_layout()
    plt.show()
