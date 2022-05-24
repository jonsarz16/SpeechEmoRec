import streamlit as st
from streamlit_option_menu import option_menu 


selected = option_menu(
    None,
    options=["Improved Algo", "Baseline", "Performance Comparison"],
    icons=["arrow-up-right-circle","arrow-repeat","app"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#a1b5d6"},
        "icon": {"color": "black", "font-size": "30px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#ffe100"},
        "nav-link-selected": {"background-color": "green"},
    }
)

if selected == "Improved Algo":
    st.title(f"You have selected {selected}")
if selected == "Baseline":
    st.title(f"You have selected {selected}")
if selected == "Performance Comparison":
    st.title(f"You have selected {selected}")