import streamlit as st
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import numpy as np 
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import random
import pandas as pd
import SessionState
#st.set_page_config(layout="wide")
class PredictionConfig(Config):
	#DETECTION_MIN_CONFIDENCE = 0.8
	#DETECTION_NMS_THRESHOLD = 0.2
    # define the name of the configuration
	NAME = "crusifixion_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 68
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
cfg = PredictionConfig()


def model_up():   
   
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model_path = 'mask_rcnn_crucifixion_cfg_0019.h5'
    model.load_weights(model_path, by_name=True)
    return model
model = model_up()

classids=["BG","crucifixion","angel","person","crown of thorns", "horse", "dragon","bird","dog","boat","cat","book",
          "sheep","shepherd","elephant","zebra","crown","tiara","camauro","zucchetto","mitre","saturno","skull",  
          "orange","apple","banana","nude","monk","lance","key of heaven", "banner","chalice","palm","sword","rooster",
          "knight","scroll","lily","horn","prayer","tree","arrow","crozier","deer","devil","dove","eagle","hands",
          "head","lion","serpent","stole","trumpet","judith","halo","helmet","shield","jug","holy shroud","god the father",
          "swan", "butterfly", "bear", "centaur","pegasus","donkey","mouse","monkey","cow"]

st.sidebar.header("Object detection API")
st.sidebar.subheader('''"Saint George on a Bike" project''')
up_file = st.sidebar.file_uploader("Choose the file",("jpg","png"))

if up_file:
    st.subheader("Detection of objects")
    my_bar = st.progress(0)
    img = load_img(up_file)
    my_bar.progress(10)
    image = img_to_array(img)
    my_bar.progress(20)
    scaled_image = mold_image(image, cfg)
    my_bar.progress(30)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    my_bar.progress(40)
    yhat = model.detect(sample, verbose=0)[0]
    fig = plt.figure(num=None, figsize=(15, 20), dpi=50, facecolor='w', edgecolor='k')
    my_bar.progress(80)
    ax = fig.add_subplot(111, aspect='equal')
    # Display the image
    ax.imshow(img)
    #Create a Rectangle patch
    for i in range(0,len(yhat['rois'])):
        #if yhat['scores'][i]>0.89:
        y1, x1, y2, x2 = yhat['rois'][i]
		# calculate width and height of the box
        width, height = x2 - x1, y2 - y1
		# create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect)
        ax.text(x1+5,y1+10,classids[yhat['class_ids'][i]],fontsize=20, color='white')
    my_bar.empty()    
    st.success('Done!')
    col1,col2 = st.beta_columns(2)
    col1.write(fig, caption=f"Processed image", use_column_width=True)
    show_pred={}
    show_pred["Bounding boxes"] = yhat["rois"].tolist()
    classes = []
    for i in yhat["class_ids"].tolist():
        classes.append(classids[i])
    show_pred["Class_ids"] = classes
    show_pred["Scores"] = yhat["scores"].tolist()     
    col2.write(pd.DataFrame.from_dict(show_pred))
    