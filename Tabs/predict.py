import streamlit as st
from web_functions import predict

def app(df, x, y):

    st.title("Halaman Prediksi")

    buying = st.text_input ('Masukan Nilai Buying')
    maint = st.text_input ('Masukan Nilai maint')
    doors = st.text_input ('Masukan Nilai doors')
    persons = st.text_input ('Masukan Nilai persons')
    lug_boot = st.text_input ('Masukan Nilai lug_boot')
    safety = st.text_input ('Masukan Nilai safety')

    features = [buying,maint,doors,persons,lug_boot,safety]

    #tombol prediksi
    if st.button("Prediksi"):
        prediction, score = predict(x, y, features)
        score = score
        st.info("Prediksi Sukses!!!")

        if (prediction== 1):
            st.warning("Kelas yang diprediksi adalah 1.")
        elif (prediction == 2):
            st.warning("Kelas yang diprediksi adalah 2.")
        elif (prediction == 3):
            st.warning("Kelas yang diprediksi adalah 3.")
        else:
            st.success("Kelas yang diprediksi adalah 4.")

        st.write("Model Digunakan Memiliki Tingkat Akurasi", (score*100),"%")