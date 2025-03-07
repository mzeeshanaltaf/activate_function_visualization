import numpy as np

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

# Activate functions formula, derivative formula, key characteristics and common use cases
# Sigmoid
sigmoid_function_label = 'Function: $\\sigma(x) = \\frac{1}{1 + e^{-x}}$'
sigmoid_derivative_label = 'Derivative: $\\sigma\'(x) = \\sigma(x)(1 - \\sigma(x))$'
sigmoid_key_characteristics = """
1. **S-shaped Curve (Non-linear)** 📈  
   - Maps input values to a smooth "S"-shaped curve between **0 and 1**.
2. **Output Range:** 🌡️  
   - The function always outputs values in the range **(0,1)**.
3. **Gradient Vanishing Issue:** ⚠️  
   - For very large or very small values of \( x \), the gradient approaches **zero**, slowing down learning in deep networks.
4. **Smooth and Differentiable:** 🔄  
   - Sigmoid is **differentiable everywhere**, making it useful in gradient-based learning.
5. **Probabilistic Interpretation:** 🎲  
   - Since the output is always between 0 and 1, it is commonly used in **binary classification** problems to represent probabilities.
6. **Zero-centered Output?** ❌  
   - **No**, sigmoid outputs are always **positive**, which can cause weight updates to be biased in a certain direction.
"""
sigmoid_use_cases = """
1. Binary Classification (Logistic Regression, Neural Networks)  
2. Output layer activation in simple networks  
3. Used in LSTM gates for regulating memory updates  
"""
# Tanh
tanh_function_label = 'Function: $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$'
tanh_derivative_label = 'Derivative: $\\tanh\'(x) = 1 - \\tanh^2(x)$'
tanh_key_characteristics = """
1. **S-shaped Curve (Non-linear)** 📈 
   - Similar to Sigmoid but symmetric around zero.
2. **Output Range:** 🌡️  
   - Outputs values in the range **(-1,1)**.
3. **Zero-centered Output:** ✅  
   - Unlike Sigmoid, **Tanh is zero-centered**, which helps in training deep networks.
4. **Gradient Vanishing Issue:** ⚠️  
   - Similar to Sigmoid, suffers from **vanishing gradients** for extreme values of \( x \).
"""
tanh_use_cases = """
1. Used in **hidden layers** of neural networks  
2. Preferred over Sigmoid when zero-centered activations are needed
"""

# ReLU
relu_function_label = 'Function: $ReLU(x) = \\max(0, x)$'
relu_derivative_label = r"Derivative: $ReLU'(x) = \{ 1 \text{ if } x > 0, 0 \text{ otherwise } \}$"
relu_key_characteristics = """
1. **Piecewise Linear & Non-linear** 🔀  
   - Outputs **0** for negative inputs and **x** for positive inputs.

2. **Output Range:** 🌡️  
   - Outputs values in the range **[0, ∞)**.

3. **Zero-centered Output?** ❌  
   - No, all negative inputs become **zero**, which may cause **dying ReLU** problem.

4. **No Vanishing Gradient Issue for Positive Inputs:** 🚀  
   - Unlike Sigmoid/Tanh, **no gradient vanishing for positive values**.

5. **Dying ReLU Problem:** ⚠️  
   - For negative inputs, gradient is **zero**, meaning neurons can become inactive.
"""
relu_use_cases = """
1. Default activation function in deep learning models  
2. Used in **CNNs, MLPs, and Transformers**
"""

# GELU
gelu_function_label = 'Function: $GELU(x) = x \\Phi(x)$, where $\\Phi(x) = \\frac{1}{2} \\left(1 + \\text{erf} \\left( \\frac{x}{\\sqrt{2}} \\right) \\right)$'
gelu_derivative_label = 'Derivative: $GELU\'(x) = \\Phi(x) + x \\phi(x)$, where $\\phi(x) = \\frac{e^{-x^2/2}}{\\sqrt{2\\pi}}$'
gelu_key_characteristics = """
1. **Smooth & Non-linear** 🔄  
   - Unlike ReLU, GELU smoothly **weights** input values instead of cutting them off.

2. **Output Range:** 🌡️  
   - Values range between **(-∞, ∞)**, making it **zero-centered**.

3. **Better than ReLU?** ✅  
   - Provides **smoother activation** and avoids sharp transitions like ReLU.

4. **Probabilistic Interpretation:** 🎲  
   - Uses **Gaussian Distribution** to determine activation.
"""
gelu_use_cases = """
1. Used in **Transformers, GPT, and BERT** models  
2. Performs better than ReLU in **deep architectures**
"""

# SwiGLU
swiglu_function_label = 'Function: $SwiGLU(x) = \\text{Swish}(x W_1) \\cdot (x W_2)$, where $\\text{Swish}(x) = x \\sigma(x)$'
swiglu_derivative_label = 'Derivative: Complex and involves product and chain rule'
swiglu_key_characteristics = """
1. **Combination of Swish & Gating Mechanism** 🔄  
   - Uses **Swish activation** and a **gated linear unit (GLU)** for better feature selection.

2. **Smooth & Non-linear** ✅  
   - Unlike ReLU, it does not abruptly cut off negative values.

3. **Element-wise Multiplication** ✖️  
   - Combines two transformed inputs, enhancing expressiveness.

4. **Better than ReLU?** 🚀  
   - Achieves **better performance** in Transformers and NLP models.
"""
swiglu_use_cases = """
1. Used in **Google’s PaLM and other Transformer models**
2. Improves training stability and efficiency in deep networks
"""

# Softmax
softmax_function_label = 'Function: $\\text{Softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j} e^{x_j}}$'
softmax_derivative_label = 'Derivative: $\\frac{\\partial \\text{Softmax}(x_i)}{\\partial x_j} = \\text{Softmax}(x_i) (\\delta_{ij} - \\text{Softmax}(x_j))$'
softmax_key_characteristics = """
1. **Transforms Inputs into Probabilities** 🎯  
   - Converts raw scores into **probability values** between **0 and 1**.
2. **Sum of Outputs Equals 1** ✅  
   - Ensures that all probabilities add up to **1**, making it ideal for classification.
3. **Exponentially Scales Inputs** 🔢  
   - Large values dominate the sum due to **exponential scaling**.
"""
softmax_use_cases = """
1. Used in **multi-class classification tasks**
2. Applied in **Neural Networks' final layer for probability prediction**
3. Used in **attention mechanisms in Transformers**
"""

# Dictionary of all activation functions
all_activation_functions = {
    "Sigmoid": {"function": sigmoid, "derivative": sigmoid_derivative,
                "label_function":sigmoid_function_label, "label_derivative":sigmoid_derivative_label,
                "key_characteristics": sigmoid_key_characteristics, "use_cases": sigmoid_use_cases},
    "Tanh": {"function": tanh, "derivative": tanh_derivative,
             "label_function":tanh_function_label, "label_derivative":tanh_derivative_label,
             "key_characteristics": tanh_key_characteristics, "use_cases": tanh_use_cases},
    "ReLU": {"function": relu, "derivative": relu_derivative,
             "label_function":relu_function_label, "label_derivative": relu_derivative_label,
             "key_characteristics": relu_key_characteristics, "use_cases": relu_use_cases},
    "GELU": {"function": gelu, "derivative": gelu_derivative,
             "label_function":gelu_function_label, "label_derivative":gelu_derivative_label,
             "key_characteristics": gelu_key_characteristics, "use_cases": gelu_use_cases},
    "SwiGLU": {"function": swiglu, "derivative": swiglu_derivative,
               "label_function":swiglu_function_label, "label_derivative":swiglu_derivative_label,
               "key_characteristics": swiglu_key_characteristics, "use_cases": swiglu_use_cases},
    "Softmax": {"function": softmax, "derivative": softmax_derivative,
                "label_function":softmax_function_label, "label_derivative":softmax_derivative_label,
                "key_characteristics": softmax_key_characteristics, "use_cases": softmax_use_cases}
}