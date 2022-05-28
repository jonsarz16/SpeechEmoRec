from streamlit_option_menu import option_menu 
from setup import *



st.set_page_config(layout="wide")
# st.header("Speech Emotion Recognition")


#baseline model
model = load_model("model.hdf5")

# improved_model
# improved_model = load_model("improved.hdf5")


selected = option_menu(
    None,
    options=["Improved Algo", "Baseline", "Performance Comparison"],
    icons=["arrow-up-right-circle","arrow-repeat","window-stack"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#a1b5d6", "display": "inline"},
        "icon": {"color": "black", "font-size": "22px"}, 
        "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#ffe100"},
        "nav-link-selected": {"background-color": "green"},}
)


if selected == "Baseline":
    modeltype = "Baseline"
    st.success(modeltype)
    done = False
    col1, col2 = st.columns(2)
    with col1:
        done = True
        run_model(modeltype)
        predict1 = st.button("Predict",key='baseline')
    with col2:
        if predict1:
            st.title("Prediction Results")
            st.success("Predicted Emotion: ILOVEYOU")

if selected == "Improved Algo":
    modeltype = "Improved"
    st.success(modeltype)
    done = False
    col1, col2 = st.columns(2)
    with col1:
        done = True
        run_model(modeltype)
        predict2 = st.button("Predict", key = 'improved')
    with col2:
        if predict2:
            st.title("Prediction Results")
            st.success("Predicted Emotion: ILOVEYOU balik kana pls")
 
 
if selected == "Performance Comparison":
    col1, col2 = st.columns(2)
     
    with col1:
        st.success("Baseline")
    
    with col2:
        st.warning("Improved Algo")
