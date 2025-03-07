import streamlit as st
from util import *

# Page configuration options
page_title = "FAQs"
page_icon = "💬"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide", initial_sidebar_state="expanded")

st.title('FAQs')

expand_all = st.toggle("Expand all", value=True)

# Display footer in the sidebar
display_footer()

faq_data = {
        'What this Application is about?':
            '<p>This app visualizes common activation functions used in neural networks along with their derivatives.</p>',
        'Which Activations are supported by this application?':
            """   <p>Following is the list of supported Activation functions by this app:</p>
                        <ul>
                            <li>Sigmoid</li>
                            <li>Tanh</li>
                            <li>ReLU</li>
                            <li>GELU</li>
                            <li>SwiGLU</li>
                            <li>Softmax</li>
                        </ul>""",
        'I want to make modification in the application. Can I get the application source code?':
            '<p>Yes, Source code of this application is available at: '
            '<a href="https://github.com/mzeeshanaltaf/activate_function_visualization">GitHub</a></p>',

    }

# Display expandable boxes for each question-answer pair
for question, answer in faq_data.items():
    with st.expander(r"$\textbf{\textsf{" + question + r"}}$", expanded=expand_all):  # Use LaTeX for bold label
        st.markdown(f'<div style="text-align: justify;"> {answer} </div>', unsafe_allow_html=True)