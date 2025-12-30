import streamlit as st

def load_neon_theme():
    st.markdown(
        """
        <style>
        body {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        .stButton>button {
            background-color: #00e5ff;
            color: black;
            border-radius: 10px;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            border: 1px solid #00e5ff;
        }
        .stSelectbox>div>div {
            border: 1px solid #00e5ff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
