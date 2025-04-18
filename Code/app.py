import streamlit as st
from PIL import Image
import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.util import img_as_ubyte
from pyfeats import glrlm_features
import pandas as pd
import pickle


def main():
    st.title(
        "Identifikasi Penyakit Cacar Monyet Menggunakan SVM Dengan Ekstraksi Fitur GLRLM"
    )

    # Model SVM
    with open("svm_model_rbf.pkl", "rb") as file:
        svm_rbf = pickle.load(file)

    st.write("")
    # Upload File
    st.write("Pilih file gambar (PNG/JPG/JPEG)")
    uploaded_file = st.file_uploader(
        "File gambar harus berfokus pada area kulit!", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        upload_col1, upload_col2, upload_col3 = st.columns(3)
        with upload_col1:
            st.write("")
        with upload_col2:
            st.image(
                image, caption="Gambar yang diunggah", use_column_width=True
            )
        with upload_col3:
            st.write("")
    
        if st.button("Proses Gambar", use_container_width=True):
            st.write("")
            # Preprocessing Data
            resized_image = preprocess_resize(image)
            grayscale_image = preprocess_grayscale(resized_image)
            blurred_image = preprocess_gaussian(grayscale_image)

            preprocess_col1, preprocess_col2, preprocess_col3 = st.columns(3)
            with preprocess_col1:
                st.image(
                    resized_image,
                    caption="Gambar di-resize (224x224)",
                    use_column_width=True,
                    channels="BGR"
                )
            with preprocess_col2:
                st.image(
                    grayscale_image,
                    caption="Gambar Grayscale",
                    use_column_width=True,
                    channels="GRAY"
                )
            with preprocess_col3:
                st.image(
                    blurred_image,
                    caption="Gambar Gaussian Blur",
                    use_column_width=True
                )

            st.write("")
            # Ekstraksi Fitur GLRLM
            st.write("Data Hasil Ekstraksi Fitur GLRLM")
            glrlm = extract_glrlm_features(blurred_image)
            st.dataframe(pd.DataFrame([glrlm], columns=[
                "Short Run Emphasis (SRE)",
                "Long Run Emphasis (LRE)",
                "Gray Level Non-Uniformity (GLN)",
                "Run Length Non-Uniformity (RLN)",
                "Run Percentage (RP)"
            ]))

            st.write("")
            # Prediksi menggunakan model SVM
            prediction = svm_rbf.predict([glrlm])
            prediction_label = "Monkeypox" if prediction == "Monkeypox" else "Normal"

            st.subheader(f"Hasil: {prediction_label}")


def preprocess_resize(image, size=(224, 224)):
    image_array = np.array(image)
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, size)
    return resized_image


def preprocess_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image


def preprocess_gaussian(image, sigma=1):
    blurred = gaussian(image, sigma=sigma)
    return img_as_ubyte(blurred)


def extract_glrlm_features(image):
    selected_features = [
        "GLRLM_ShortRunEmphasis",
        "GLRLM_LongRunEmphasis",
        "GLRLM_GrayLevelNo-Uniformity",
        "GLRLM_RunLengthNonUniformity",
        "GLRLM_RunPercentage",
    ]

    mask = np.ones_like(image, dtype=np.uint8)
    features, labels = glrlm_features(image, mask)

    selected_features_indices = [labels.index(f) for f in selected_features]
    filtered_features = features[selected_features_indices]
    return filtered_features


if __name__ == "__main__":
    main()
