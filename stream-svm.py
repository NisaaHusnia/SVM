import pickle
import streamlit as st
import pandas as pd

# Fungsi untuk memuat model dari file pickle
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

# Konfigurasi halaman Streamlit dengan tampilan terang
st.set_page_config(page_title="SVM Prediction", page_icon="📊", layout="wide")

# Header dan Deskripsi dengan styling
st.markdown("""
    <style>
        .title {
            font-size: 48px;
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .description {
            font-size: 18px;
            color: #333;
            margin-top: 10px;
            text-align: center;
        }
        .header-text {
            font-size: 24px;
            color: #FF6347;
            text-align: center;
            font-weight: bold;
            margin-top: 20px;
        }
        .success-text {
            font-size: 24px;
            color: #4CAF50;
            font-weight: bold;
        }
        .container-box {
            border: 2px solid #4CAF50;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">🎯 SVM Predictor</p>', unsafe_allow_html=True)

# Deskripsi aplikasi
st.markdown('<p class="description">Pilih dataset yang ingin diprediksi. Masukkan nilai yang sesuai dengan inputan yang disediakan dan klik <strong>Prediksi</strong> untuk melihat hasilnya.</p>', unsafe_allow_html=True)

# Pilihan dataset
datasets = {
    "Fish Dataset": {
        "model": "svm_fish_model.pkl",
        "sample_data": "fish_data.csv",
        "input_columns": ["length", "weight", "w_l_ratio"],
        "species_labels": [
            'Anabas testudineus', 'Coilia dussumieri', 'Otolithoides biauritus',
            'Otolithoides pama', 'Pethia conchonius', 'Polynemus paradiseus',
            'Puntius lateristriga', 'Setipinna taty', 'Sillaginopsis panijus'
        ]
    },
    "Fruit Dataset": {
        "model": "svm_fruit_model.pkl",
        "sample_data": "fruit.xlsx",
        "input_columns": ["diameter", "weight", "red", "green", "blue"],
        "species_labels": ['grapefruit', 'orange']
    },
    "Pumpkin Dataset": {
        "model": "svm_pumpkin_model.pkl",
        "sample_data": "Pumpkin_Seeds_Dataset.xlsx",
        "input_columns": [
            "Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length",
            "Convex_Area", "Equiv_Diameter", "Eccentricity", "Solidity", 
            "Extent", "Roundness", "Aspect_Ration", "Compactness"
        ],
        "species_labels": ['Ürgüp Sivrisi','Çerçevelik']
    }
}

# Dropdown untuk memilih dataset
dataset_choice = st.selectbox("Pilih Dataset", list(datasets.keys()), index=0)

# Memuat model dan data sampel
selected_dataset = datasets[dataset_choice]
model_path = selected_dataset["model"]
sample_data_path = selected_dataset["sample_data"]
input_columns = selected_dataset["input_columns"]
species_labels = selected_dataset.get("species_labels", None)

# Memulai container
with st.container():
    # Menampilkan Model yang Dipilih dan Memuat Model
    st.write(f"**Model yang dipilih:** {model_path}")
    model = load_model(model_path)

    if model is not None:
        st.success("✔️ Model berhasil dimuat!")

        # Memuat data sampel untuk prediksi
        st.markdown('<p class="header-text">Contoh data untuk prediksi:</p>', unsafe_allow_html=True)
        if sample_data_path.endswith('.csv'):
            sample_data = pd.read_csv(sample_data_path)
        elif sample_data_path.endswith('.xlsx'):
            sample_data = pd.read_excel(sample_data_path)
        else:
            st.error("Format file data tidak didukung.")
            sample_data = pd.DataFrame()  # Set data kosong jika format tidak dikenal

        # Menampilkan data sampel dalam kotak
        with st.expander("Klik untuk melihat contoh data"):
            st.dataframe(sample_data, width=800)

        # Input untuk prediksi
        st.markdown('<p class="header-text">Masukkan data untuk prediksi:</p>', unsafe_allow_html=True)
        input_data = {}

        for col in input_columns:
            input_data[col] = st.number_input(f"Masukkan nilai untuk **{col}**", value=sample_data[col].iloc[0] if col in sample_data.columns else 0.0)

        # Sinkronkan kolom input dengan kolom yang digunakan oleh model
        data_for_prediction = pd.DataFrame([input_data], columns=input_columns)

        # Tombol untuk memulai prediksi
        st.markdown("---")
        if st.button("🔮 **Prediksi**", key="predict_button"):
            if not data_for_prediction.empty:
                try:
                    prediction = model.predict(data_for_prediction)

                    # Menampilkan hasil prediksi dengan nama spesies untuk dataset dengan label string
                    if species_labels:
                        # Verifikasi tipe data prediksi dan pastikan memetakan dengan benar
                        if isinstance(prediction[0], int):
                            predicted_species = species_labels[int(prediction[0])]
                        elif isinstance(prediction[0], str):
                            predicted_species = prediction[0]
                        else:
                            predicted_species = "Unknown"
                        
                        st.markdown(f"<p class='success-text'>**Hasil Prediksi untuk {dataset_choice}:** {predicted_species} (species {prediction[0]})</p>", unsafe_allow_html=True)
                    else:
                        st.write(f"### Hasil Prediksi untuk {dataset_choice}: {prediction[0]}")
                except IndexError:
                    st.error("Hasil prediksi berada di luar indeks label yang tersedia.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam prediksi: {e}")
            else:
                st.error("Data untuk prediksi kosong!")
    else:
        st.error("Model gagal dimuat.")
