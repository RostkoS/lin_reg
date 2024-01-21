import streamlit as st
import numpy as np
from prediction import predict
st.title('LinearRegression')
st.header("Features")
col1, col2 = st.columns(2)
with col1:
  st.text('Weight')
  w_w = st.slider('WholeWeight', 0.1 , 3.0, 0.01)
  s_w = st.slider('ShuckedWeight', 0.1 , 3.0, 0.01)
  v_w = st.slider('VisceraWeight', 0.1 , 3.0, 0.01)
  s_w1 = st.slider('ShellWeight', 0.1 , 3.0, 0.01)
with col2:
  d = st.slider('Diameter', 0.01 , 1.0, 0.01)
  h = st.slider('Height', 0.01 , 1.0, 0.01)
  r = st.slider('Rings',5 , 20.0, 1)
if st.button("Predict LongestShell"):
  result = predict(np.array([[d, h, w_w, s_w, v_w, s_w1, r]]))
st.text(result[0])
