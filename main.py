import streamlit as st
from util import *

# Page title of the application
page_title = "Activ8Viz"
page_icon = "📈"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

# Application Title and description
st.title(f'{page_title}{page_icon}')
st.write('***:blue[Visualize & Explore Neural Network Activations! 📈✨]***')
st.write("""
*Activ8Viz is your go-to web app for exploring and visualizing activation functions and their derivatives used in 
Neural Networks. 🧠📈 From Sigmoid to SwiGLU, dive into the math behind AI with interactive plots and understand how 
they shape deep learning models. Enhance your AI intuition with dynamic visualizations! Perfect for learners, 
researchers, and AI enthusiasts! 🚀🔍*
""")
# Display footer in the sidebar
display_footer()

# Options for activation functions
options = ["Sigmoid", "Tanh", "ReLU", "GELU", "SwiGLU", "Softmax"]
st.subheader('Select Activation Function', divider='gray')
selection = st.pills('Select the Activation Function', options, selection_mode="multi", default=options,
                     label_visibility="collapsed")
plot_derivative = st.checkbox("Plot Derivative", value=True)

if not selection:
    st.warning("Please select at least one activation function.", icon=":material/warning:")

else:
    with st.spinner('Processing ...', show_time=True):
        num_plots = len(selection)
        max_cols = 3
        for i in range(0, num_plots, max_cols):
            cols = st.columns(min(max_cols, num_plots - i))
            for j, col in enumerate(cols):
                if i + j < num_plots:
                    with col:
                        fig = create_activation_plot(selection[i+j], plot_derivative)
                        st.pyplot(fig)




