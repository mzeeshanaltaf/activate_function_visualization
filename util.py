import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# Define activation functions and their derivatives
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid activation function"""
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of hyperbolic tangent activation function"""
    return 1 - np.tanh(x)**2

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU activation function"""
    return np.where(x > 0, 1, 0)

def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """Approximate derivative of GELU activation function"""
    # Approximation
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf

def swiglu(x):
    """SwiGLU activation function (simplified for 1D)"""
    # For simplicity, we'll use the same input for both parts
    # SwiGLU(x,y) = x * sigmoid(beta * y)
    beta = 1.0  # Default beta value
    return x * sigmoid(beta * x)

def swiglu_derivative(x):
    """Approximate derivative of SwiGLU activation function (simplified for 1D)"""
    beta = 1.0
    sig = sigmoid(beta * x)
    return sig + x * beta * sig * (1 - sig)

def softmax(x):
    """Softmax activation function (simplified for 1D visualization)"""
    # For visualization purposes, we'll just show a shifted exponential
    # since proper softmax requires multiple inputs
    return np.exp(x) / (1 + np.exp(x))

def softmax_derivative(x):
    """Simplified derivative of softmax (just for visualization)"""
    # This is a simplification for visualization
    soft = softmax(x)
    return soft * (1 - soft)

# Sigmoid
sigmoid_function_label = 'Sigmoid: $\\sigma(x) = \\frac{1}{1 + e^{-x}}$'
sigmoid_derivative_label = 'Sigmoid Derivative: $\\sigma\'(x) = \\sigma(x)(1 - \\sigma(x))$'

# Tanh
tanh_function_label = 'Tanh: $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$'
tanh_derivative_label = 'Tanh Derivative: $\\tanh\'(x) = 1 - \\tanh^2(x)$'

# ReLU
relu_function_label = 'ReLU: $ReLU(x) = \\max(0, x)$'
relu_derivative_label = r"ReLU Derivative: $ReLU'(x) = \{ 1 \text{ if } x > 0, 0 \text{ otherwise } \}$"

# GELU
gelu_function_label = 'GELU: $GELU(x) = x \\Phi(x)$, where $\\Phi(x) = \\frac{1}{2} \\left(1 + \\text{erf} \\left( \\frac{x}{\\sqrt{2}} \\right) \\right)$'
gelu_derivative_label = 'GELU Derivative: $GELU\'(x) = \\Phi(x) + x \\phi(x)$, where $\\phi(x) = \\frac{e^{-x^2/2}}{\\sqrt{2\\pi}}$'

# SwiGLU
swiglu_function_label = 'SwiGLU: $SwiGLU(x) = \\text{Swish}(x W_1) \\cdot (x W_2)$, where $\\text{Swish}(x) = x \\sigma(x)$'
swiglu_derivative_label = 'SwiGLU Derivative: Complex and involves product and chain rule'

# Softmax
softmax_function_label = 'Softmax: $\\text{Softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j} e^{x_j}}$'
softmax_derivative_label = 'Softmax Derivative: $\\frac{\\partial \\text{Softmax}(x_i)}{\\partial x_j} = \\text{Softmax}(x_i) (\\delta_{ij} - \\text{Softmax}(x_j))$'


all_activation_functions = {
    "Sigmoid": {"function": sigmoid, "derivative": sigmoid_derivative, "label_function":sigmoid_function_label, "label_derivative":sigmoid_derivative_label},
    "Tanh": {"function": tanh, "derivative": tanh_derivative, "label_function":tanh_function_label, "label_derivative":tanh_derivative_label},
    "ReLU": {"function": relu, "derivative": relu_derivative, "label_function":relu_function_label, "label_derivative": relu_derivative_label},
    "GELU": {"function": gelu, "derivative": gelu_derivative, "label_function":gelu_function_label, "label_derivative":gelu_derivative_label},
    "SwiGLU": {"function": swiglu, "derivative": swiglu_derivative, "label_function":swiglu_function_label, "label_derivative":swiglu_derivative_label},
    "Softmax": {"function": softmax, "derivative": softmax_derivative, "label_function":softmax_function_label, "label_derivative":softmax_derivative_label}
}

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