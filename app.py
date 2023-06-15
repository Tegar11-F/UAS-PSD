import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas as pd
import pickle
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.metrics import mean_absolute_percentage_error

# st.sidebar.title("Selamat Datang!")
# st.sidebar.write(
#     "Di Website Prediksi Saham Perusahaan Bukalapak.")
page1, page2, page3, page4 = st.tabs(
    ["Data", "Preprocessing", "Modelling", "Implementasi"])

with page1:
    st.title("Prediksi Saham AMD")
    st.write("Website ini bertujuan untuk memprediksi saham AMD yang akan datang, Dataset yang di gunakan di peroleh dari yahoo")
    st.header("Data Sample")
    st.text("""
    menggunakan kolom:
    * Open: Harga pembukaan (open price)
    * High: Harga tertinggi (high price)
    * Low: Harga terendah (low price)
    * Close: Harga penutupan (closing price)
    * Adj Close: Harga penutupan yang disesuaikan (adjusted closing price)
    * Volume: Volume perdagangan adalah jumlah total saham yang diperdagangkan pada hari tersebut.

    """)
    st.caption('link datasets : https://finance.yahoo.com/quote/AMD/history?p=AMD')
    df_data = pd.read_csv("https://raw.githubusercontent.com/Tegar11-F/Data-Set/main/AMD.csv")
    st.dataframe(df_data.head())

with page2:
    st.title("Preprocessing")

    st.write( "time series prediction, disini kami menggunakan 5 prediction")
    st.write( "Min Max Scaler digunakan untuk mengubah data numerik menjadi data yang memiliki range 0 sampai 1")

    df_passenger= df_data['Close']

    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
        # find the end of this pattern
            end_ix = i + n_steps
        # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
        # gather input and output parts of the pattern
            # print(i, end_ix)
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    
    n_steps = 5
    X, y = split_sequence(df_passenger, n_steps)

    # column names to X and y data frames
    df_X = pd.DataFrame(X, columns=['t-'+str(i) for i in range(n_steps-1, -1,-1)])
    df_y = pd.DataFrame(y, columns=['t+1 (prediction)'])

    # concat df_X and df_y
    df = pd.concat([df_X, df_y], axis=1)

    df.head(3)

    scaler= MinMaxScaler()
    X_norm= scaler.fit_transform(df_X)
    X_norm

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)



with page3:
    st.subheader("Modelling Data")

    st.write('Setelah melalui proses preprocessing data, langkah berikutnya adalah pembentukan model (Modelling). Silahkan pilih model yang ingin digunakan, kemudian tekan tombol "Modelling" untuk memulai proses modelling.')

    with st.form(key="Form3"):
        model = st.selectbox(
            "Pilih Model",
            (
                "K-Nearest Neighbors",
                "Decision Tree",
                "Decision Tree",
            ),
        )
        submitted2 = st.form_submit_button(label="Modelling")

    if submitted2:
        st.caption('link datasets : https://finance.yahoo.com/quote/AMD/history?p=AMD')
        df_data = pd.read_csv("https://raw.githubusercontent.com/Tegar11-F/Data-Set/main/AMD.csv")
        st.dataframe(df_data.head())
        def split_sequence(sequence, n_steps):
                X, y = list(), list()
                for i in range(len(sequence)):
                # find the end of this pattern
                    end_ix = i + n_steps
                # check if we are beyond the sequence
                    if end_ix > len(sequence)-1:
                        break
                # gather input and output parts of the pattern
                    # print(i, end_ix)
                    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                    X.append(seq_x)
                    y.append(seq_y)
                return array(X), array(y)
            
        n_steps = 5
        X, y = split_sequence(df_passenger, n_steps)

        # column names to X and y data frames
        df_X = pd.DataFrame(X, columns=['t-'+str(i) for i in range(n_steps-1, -1,-1)])
        df_y = pd.DataFrame(y, columns=['t+1 (prediction)'])

        # concat df_X and df_y
        df = pd.concat([df_X, df_y], axis=1)

        df.head(3)

        scaler= MinMaxScaler()
        X_norm= scaler.fit_transform(df_X)
        X_norm

        X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)
        if model == "K-Nearest Neighbors":
            model_knn = KNeighborsRegressor(n_neighbors=5)
            model_knn.fit(X_train, y_train)

            filename_knn = 'model_knn.sav'
            pickle.dump(model_knn, open(filename_knn, 'wb'))
            y_pred=model_knn.predict(X_test)

            mape = mean_absolute_percentage_error(y_test, y_pred)

    elif model == "Decision Tree":
        model_dt = DecisionTreeRegressor()
        model_dt.fit(X_train, y_train)



        dt_mms = 'model_dt.sav'
        pickle.dump(model_dt, open(dt_mms, 'wb'))

        # Prediksi pada data test
        y_pred = model_dt.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)

    elif model == "Decision Tree":
        max_iter = 100
        tolerance = 0.0001

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
            model = MLPRegressor(hidden_layer_sizes=(100, 100),max_iter=max_iter,tol=tolerance)

            # train model
            model.fit(X_train, y_train)

            # prediksi data test
            y_MLP = model.predict(X_test)
        mlp = 'model_tree.sav'
        pickle.dump(model, open(mlp, 'wb'))
        y_pred = model.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)

    else:
        st.write("Tidak ada preprocessing yang dipilih")

    st.subheader("Akurasi Model")
    st.write("Berikut adalah akurasi model yang anda pilih:")
    st.write("Mean Absolute Percentage Error: ", round(100 * mape, 2), "%")
