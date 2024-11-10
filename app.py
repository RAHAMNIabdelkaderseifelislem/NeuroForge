import streamlit as st
from ui.home_template import home_template
from ui.builder_template import builder_template

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Builder"])

    if page == "Home":
        home_template()
    elif page == "Builder":
        builder_template()

if __name__ == "__main__":
    main()
