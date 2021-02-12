import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from PIL import Image

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def rgb2gray(rgb):
    ''' Берётся среднее трёх цветов RGB'''
    tile = np.tile(np.c_[0.333, 0.333, 0.333], reps=(rgb.shape[0],rgb.shape[1],1))
    return np.sum(tile * rgb, axis=2)

def preprocess_model(img, n_components):
    svd_model = TruncatedSVD(n_components=n_components)
    X_svd = svd_model.fit_transform(img)
    X_restored = svd_model.inverse_transform(X_svd)
    return X_restored



def main():
    st.title('SVD Decomposition Example')
    menu = ['Images']
    choice = st.sidebar.selectbox('Images', menu)
    if choice == 'Images':
        image_file = st.file_uploader('Upload Images', type=['png', 'jpg', 'jpeg'])
        if image_file is not None:
            img_details = {'filename': image_file.name, 'filetype': image_file.type, 
                            'filesize':image_file.size}
            st.write(img_details)
            fig = px.imshow(load_image(image_file))
            st.write(fig)
            img_gray = rgb2gray(mpimg.imread(image_file))
            slider = st.slider('SVD Decomposition', min_value=5, max_value=50, step=5, value=25)
            st.write(px.imshow(preprocess_model(img_gray, slider), binary_string=True))




          
            











if __name__ == '__main__':
    main()
