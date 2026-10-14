import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# FIX SEED (Ổn định kết quả dự báo)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Cấu hình trang Streamlit
st.set_page_config(page_title="Dự Báo Giá Cổ Phiếu bằng LSTM", page_icon="📈", layout="wide")

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu bằng Mô Hình LSTM")
st.write("Dự báo xu hướng giá cổ phiếu trong ngắn hạn và tính toán phần trăm *Lãi / Lỗ* kỳ vọng.")

# --- THANH CẤU HÌNH (SIDEBAR) ---
st.sidebar.header("⚙️ Cấu Hình Dự Báo")
ticker_input = st.sidebar.text_input("Mã chứng khoán:", value="VNM")
start_date = st.sidebar.date_input("Ngày bắt đầu lấy dữ liệu:", datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc:", datetime.date.today())

n_days = st.sidebar.slider("Số ngày muốn dự báo tương lai:", min_value=1, max_value=30, value=10)
epochs = st.sidebar.slider("Số Epochs huấn luyện:", min_value=5, max_value=30, value=10)

btn_predict = st.sidebar.button("🚀 Bắt Đầu Dự Báo", type="primary")

# Tự động thêm đuôi .VN nếu nhập mã Việt Nam (3 ký tự)
ticker = ticker_input.strip().upper()
if len(ticker) == 3 and not ticker.endswith(".VN"):
    ticker = ticker + ".VN"

if btn_predict:
    with st.spinner(f"Đang tải dữ liệu {ticker} và huấn luyện mô hình LSTM... Vui lòng đợi trong giây lát!"):
        # 1. Tải dữ liệu từ Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"❌ Không tìm thấy dữ liệu cho mã '{ticker}'. Vui lòng kiểm tra lại!")
        else:
            # Lấy cột giá đóng cửa Close
            if isinstance(df.columns, pd.MultiIndex):
                data = df['Close'][[ticker]].values
            else:
                data = df[['Close']].values
                
            st.success(f"Đã tải thành công **{len(data)}** phiên giao dịch của **{ticker}**!")

            # 2. Chia tập Train (80%) / Test (20%) & Chuẩn hóa dữ liệu
            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            test_data = data[train_size:]

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_data)
            test_scaled = scaler.transform(test_data)

            # Hàm tạo chuỗi thời gian (window_size = 60)
            def create_sequences(data_seq, window_size=60):
                X, y = [], []
                for i in range(window_size, len(data_seq)):
X.append(data_seq[i-window_size:i])
                    y.append(data_seq[i])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train_scaled)
            test_combined = np.vstack((train_scaled[-60:], test_scaled))
            X_test, y_test = create_sequences(test_combined)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            val_size = int(0.1 * len(X_train))
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_final = X_train[:-val_size]
            y_train_final = y_train[:-val_size]

            # 3. Xây dựng mô hình LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # Train mô hình
            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # 4. Đánh giá trên tập Test
            test_pred = model.predict(X_test, verbose=0)
            test_pred = scaler.inverse_transform(test_pred)
            y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

            rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
            mae = mean_absolute_error(y_test_real, test_pred)
            r2 = r2_score(y_test_real, test_pred)

            # 5. Dự báo n ngày tiếp theo (Cuốn chiếu)
            last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
            last_60_days = last_60_days.reshape(1, 60, 1)

            future_predictions = []
            for i in range(n_days):
                next_pred = model.predict(last_60_days, verbose=0)
                future_predictions.append(next_pred[0, 0])
                last_60_days = np.append(
                    last_60_days[:, 1:, :],
                    next_pred.reshape(1, 1, 1),
                    axis=1
                )

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)

            # Ngày làm việc cho n ngày dự báo
            last_date = df.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

            # Tính toán các chỉ số tóm tắt Lãi / Lỗ
            last_real_price = float(data[-1][0])
final_pred_price = float(future_predictions[-1][0])
            diff_price = final_pred_price - last_real_price
            pct_change = (diff_price / last_real_price) * 100

            # --- HIỂN THỊ KẾT QUẢ TRÊN GIAO DIỆN ---
            
            # KHỐI 1: TÓM TẮT LÃI / LỖ
            st.markdown("---")
            st.subheader("🎯 Tóm Tắt Dự Báo Lãi / Lỗ")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Giá đóng cửa gần nhất", f"{last_real_price:,.2f}")
            m2.metric(f"Giá dự báo ngày thứ {n_days}", f"{final_pred_price:,.2f}")
            m3.metric("Mức chênh lệch", f"{diff_price:+,.2f}")
            m4.metric("Dự báo % Lãi / Lỗ", f"{pct_change:+.2f}%", delta=f"{pct_change:+.2f}%")

            # KHỐI 2: BẢNG DỰ BÁO CHI TIẾT
            st.markdown("---")
            st.subheader(f"📋 Bảng Dự Báo Giá {n_days} Ngày Tới")
            
            df_future = pd.DataFrame({
                "Ngày dự báo": [d.strftime('%d/%m/%Y') for d in future_dates],
                "Giá dự báo": future_predictions.flatten(),
                "% Thay đổi so với giá gần nhất": [
                    f"{((p - last_real_price) / last_real_price * 100):+.2f}%" for p in future_predictions.flatten()
                ]
            })
            
            st.dataframe(
                df_future.style.format({"Giá dự báo": "{:,.2f}"}),
                use_container_width=True
            )

            # KHỐI 3: BIỂU ĐỒ TRỰC QUAN HÓA (TABS)
            st.markdown("---")
            st.subheader("📊 Biểu Đồ Trực Quan Hóa")
            
            tab1, tab2 = st.tabs(["Biểu đồ Tương tác (Plotly)", "Hàm Mất Mát (Train vs Val Loss)"])

            with tab1:
                fig = go.Figure()

                # Đường giá thực tế tập test
                test_dates = df.index[train_size:]
                fig.add_trace(go.Scatter(
                    x=test_dates, y=y_test_real.flatten(),
                    mode='lines', name='Giá thực tế (Test set)',
                    line=dict(color='#1f77b4', width=2)
                ))

                # Đường dự báo tập test
                fig.add_trace(go.Scatter(
                    x=test_dates, y=test_pred.flatten(),
                    mode='lines', name='Dự báo (Test set)',
                    line=dict(color='#ff7f0e', width=1.5, dash='dash')
                ))

                # Đường n ngày tới
                fig.add_trace(go.Scatter(
                    x=future_dates, y=future_predictions.flatten(),
                    mode='lines+markers', name=f'Dự báo {n_days} ngày tới',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=8)
                ))

                fig.update_layout(
                    title=f"So sánh Giá Thực tế vs Dự báo & {n_days} Ngày Tới của mã {ticker}",
xaxis_title="Thời gian / Chỉ số phiên",
                    yaxis_title="Giá cổ phiếu",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'], mode='lines+markers', name='Train Loss',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'], mode='lines+markers', name='Validation Loss',
                    line=dict(color='#d62728', width=2)
                ))
                fig_loss.update_layout(
                    title="Đường cong Hàm Mất Mát qua các Epochs",
                    xaxis_title="Epoch",
                    yaxis_title="Loss (Mean Squared Error)",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            # KHỐI 4: CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH
            st.markdown("---")
            st.subheader("📐 Chỉ Số Đánh Giá Mô Hình (Model Metrics)")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.4f}")
            c2.metric("MAE (Mean Absolute Error)", f"{mae:,.4f}")
            c3.metric("R² Score", f"{r2:.4f}")

