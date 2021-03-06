from streamlit_option_menu import option_menu 
from setup import *
import tensorflow as tf
import pandas as pd

pd.set_option('precision', 2)
# pd.reset_option('display.float_format')

# st.header("Speech Emotion Recognition")
local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# with st.sidebar:
#     st.write("Speech Emotion")
#     agree = st.checkbox('Use Test file')
# if agree:
#     test_audio = "YAF_back_angry.wav"
#     data_visual_improved(test_audio)

# else:
selected = option_menu(
    None,
    options=["Improved Algo", "Baseline", "Performance Comparison"],
    icons=["arrow-up-right-circle","arrow-repeat","window-stack"],
    menu_icon="cast",
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#3CAEA3", "display": "inline"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#ffe100"},
        "nav-link-selected": {"background-color": "#0F2557"},}
)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotions1 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
x = [[ '😡','angry'], [ '🤮','disgust'], ['🥶','fear'],['😀','happy'], ['🤴','neutral'], ['😢','sad'],['😱','surprise']]
                
hehe = ['😡','🤮','🥶','😀','🤴','😢','😱']
if selected == "Improved Algo":
    st.warning("to be continued")
    # col1, col2 = st.columns(2)
    # container1 = st.empty()
    
    # with col1:
    #     file_audio1 = container1.file_uploader("", type=['mp3','wav'])
        
    #     if file_audio1 is not None:
    #         data_visual_improved(file_audio1)
    #         predict2 = container1.button("Predict", key = 'improved')

    # with col2:
    #     try:
    #         if predict2:
                
    #             st.title("Prediction Results X")
                       
    #             container1.empty()
    #     except:
    #         pass
        

if selected == "Baseline":
    col1, col2 = st.columns(2)
    container2 = st.empty()
    
    with col1:
        file_audio2 = container2.file_uploader("", type=['mp3','wav'])
        
        if file_audio2 is not None:
            data_visual_baseline(file_audio2)
            predict2 = container2.button("Predict", key = 'improved')
           

    with col2:
        try:
            if predict2:
                container2.empty()
                st.title("Prediction Results")
                result = classify('melspecs.png')
                score = tf.nn.softmax(result[0])
             
                # st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
                # .format(emotions[np.argmax(result)], 100 * np.max(result)))
              
                st.write("Predicted emotion:  **{}**."
                .format(emotions[np.argmax(score)].upper()))
                
                
                var = score.numpy() * 100
                
                
                # var = tf.convert_to_tensor(result).numpy()
                # z = np.swapaxes(var,0,1)
                
                # df = pd.DataFrame(emotions, columns=["Predicted emotion"])
                df = pd.DataFrame(x, columns=["","Predicted emotion"])
                df['Percentage'] = var
                df['Percentage'] = df['Percentage'].apply(lambda x: float("{:,.2f}".format(x)))
                df['.'] = "%"
                
                df = df.style.background_gradient()
                st.table(df)
            
        except:
            pass

if selected == "Performance Comparison":
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("Baseline")
    
    with col2:
        st.warning("Improved Algo")





