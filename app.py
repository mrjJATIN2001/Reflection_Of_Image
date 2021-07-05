import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
import cv2 as cv
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
from keras.models import load_model

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""Transformation""")
file= st.file_uploader("Please upload image", type=("jpg", "png"))

from  PIL import Image, ImageOps
def reflection_x(image):
  #img = image.load_img(image, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)
  rows, cols, dim = image.shape
  M = np.float32([[ 1,  0, cols],
                  [ 0, -1, rows],
                  [ 0,  0,    1]])
  re_img = cv.warpPerspective(image,M,(int(cols*2),int(rows*2)))
  st.image(re_img,caption='X-Reflected Image.', use_column_width=True)
  return re_img

def reflection_y(image):
  #img = image.load_img(image, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)
  rows, cols, dim = image.shape
  M = np.float32([[-1,  0, cols],
                  [ 0,  1, rows],
                  [ 0,  0,    1]])
  re_img = cv.warpPerspective(image,M,(int(cols*2),int(rows*2)))
  st.image(re_img,caption='Y-Reflected Image.', use_column_width=True)
  return re_img

if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Reflection X-axis"):
  result=reflection_x(image)
if st.button("Reflection Y-axis"):
  result=reflection_y(image)

  
if st.button("About"):
  st.header(" Jatin Tak")
  st.subheader("Student, Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)