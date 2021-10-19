import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt


from utils import data_preprocessing as dp


if __name__ == '__main__':

    model = load_model('model/v2_jacques.h5')

    explanation_expander = st.beta_expander('Some explanation on lesions')
    with explanation_expander:
        st.write('try this')

    st.title('Skin cancer detector')

    uploaded_file = st.file_uploader('Upload an lesion image')
    if uploaded_file is not None:
        valid_file = False
        try:
            img = Image.open(uploaded_file)
            valid_file = True

            # plot
            # fig, ax = plt.subplots()
            # plt.imshow(img)
            # st.pyplot(fig)

            prepared_image = dp.prepare_one_image_no_tl(img)

            pred_array = model.predict(prepared_image)
            st.write(pred_array)
            pred_index = np.argmax(pred_array)

            classes = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
            cancerous_classes = ['mel', 'bkl', 'bcc']  # bkl = warning

            predicted_class = classes[pred_index]

            st.write('This is the predicted class:', predicted_class)
            if predicted_class in cancerous_classes:
                st.write('Bad news')
            else:
                st.write('Good news')
        except:
            st.write('Sorry but we do not understand the uploaded file.'
                     'Please make sure to upload an image file.')

    hide_st_style = """ <style> footer {visibility: hidden;} </style> """
    st.markdown(hide_st_style, unsafe_allow_html=True)
