import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
# import platform

# plt = platform.system()
# if plt == 'Linux':
# pathlib.WindowsPath = pathlib.PosixPath


# title
st.title("Transportni klassifikatsiya qiluvchi model")

file = st.file_uploader('Rasm yukash', type=['png', 'svg', 'jpg', 'gif', 'jpeg'])
if file:
    st.image(file)
    img = PILImage.create(file)


    model = load_learner('transport_model.pkl')

    pred, pred_id, probs = model.predict(img)
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimolligi: {probs[pred_id]*100:.1f}%')


    # plotting
    figure = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(figure)