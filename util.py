import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from activation_functions import *

@st.cache_resource
def create_activation_plot(title, plot_derivative):
    """
    Create a plot of an activation function and its derivative
    with dynamically adjusted size based on number of columns
    """
    figsize = (8, 8)
    x = np.linspace(-5, 5, 100)

    # Handle special case for ReLU to avoid interpolation artifacts
    if title == "ReLU":
        # Add points very close to zero to capture the discontinuity
        extra_points = np.array([-0.001, 0, 0.001])
        x = np.sort(np.concatenate([x, extra_points]))

    activation_fn = all_activation_functions[title]["function"]
    y_activation = activation_fn(x)
    if plot_derivative:
        derivative_fn = all_activation_functions[title]["derivative"]
        y_derivative = derivative_fn(x)

    fig, ax = plt.subplots(figsize=figsize)

    function_label = all_activation_functions[title]["label_function"]
    derivative_label = all_activation_functions[title]["label_derivative"]

    # Plot activation function
    ax.plot(x, y_activation, 'b-', linewidth=2, label=function_label)

    if plot_derivative:
        # Plot derivative
        ax.plot(x, y_derivative, 'r-', linewidth=2, label=derivative_label)

    # Add grid, legend, and labels
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Adjust font sizes based on figure size

    title_size = 16
    label_size = 14
    legend_size = 12

    ax.set_xlabel('x', fontsize=label_size)
    ax.set_ylabel('y', fontsize=label_size)
    ax.set_title(title, fontsize=title_size)
    ax.legend(fontsize=legend_size, loc='upper left')

    # Set appropriate y-axis limits based on the function
    if activation_fn == relu:
        ax.set_ylim(-0.1, max(5, np.max(y_activation) * 1.1))
    elif activation_fn in [sigmoid, softmax]:
        ax.set_ylim(-0.1, 1.1)
    elif activation_fn == tanh:
        ax.set_ylim(-1.1, 1.1)
    else:
        if plot_derivative:
            # Dynamically set limits for other functions
            y_min = min(np.min(y_activation), np.min(y_derivative)) * 1.1
            y_max = max(np.max(y_activation), np.max(y_derivative)) * 1.1
        else:
            # Dynamically set limits for other functions
            y_min = min([np.min(y_activation)]) * 1.1
            y_max = max([np.max(y_activation)]) * 1.1

        # Add some padding
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()
    return fig

@st.dialog("About Activation Function", width='large')
def more_info_dialog(title):
    st.write("**Name:** ", "$"+f"{title}"+"$")
    st.write("**Formula:** ", all_activation_functions[title]["label_function"].split(":")[-1])
    st.write("**Derivative:** ", all_activation_functions[title]["label_derivative"].split(":")[-1])
    st.write("**Key Characteristics:** ", all_activation_functions[title]["key_characteristics"])
    st.write("**Common Use Cases:** ", all_activation_functions[title]["use_cases"])

def display_footer():
    footer = """
    <style>
    /* Ensures the footer stays at the bottom of the sidebar */
    [data-testid="stSidebar"] > div: nth-child(3) {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
    }

    .footer {
        color: grey;
        font-size: 15px;
        text-align: center;
        background-color: transparent;
    }
    </style>
    <div class="footer">
    Made with ❤️ by <a href="mailto:zeeshan.altaf@gmail.com">Zeeshan</a>.
    </div>
    """
    st.sidebar.markdown(footer, unsafe_allow_html=True)