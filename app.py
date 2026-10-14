import random
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

# -------------------- Cấu hình --------------------
st.set_page_config(
    page_title="Stock Forecast - LSTM",
    page_icon="📈",
    layout="wide",
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.title("📈 Dự báo giá cổ phiếu bằng LSTM")
st.caption(
    "Ứng dụng học tập minh họa dự báo giá đóng cửa từ dữ liệu Yahoo Finance. "
    "Kết quả không phải khuyến nghị đầu tư."
)

# -------------------- Hàm dữ liệu --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return pd.DataFrame()

    # Tương thích với các phiên bản yfinance trả về MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)]
        else:
            close_cols = [col for col in df.columns if col[0] == "Close"]
            if not close_cols:
                return pd.DataFrame()
            close = df[close_cols[0]]
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df["Close"]

    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    result.index = pd.to_datetime(result.index)
    result = result.dropna().sort_index()
    result.index.name = "Date"
    return result


def create_sequences(values: np.ndarray, window_size: int = 60):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def build_model(window_size: int = 60):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# -------------------- Thanh bên --------------------
st.sidebar.header("⚙️ Thiết lập")
ticker = st.sidebar.text_input("Mã cổ phiếu Yahoo Finance", value="VNM.VN").strip().upper()
start_date = st.sidebar.date_input("Ngày bắt đầu", value=date(2018, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc dữ liệu", value=date.today())
forecast_days = st.sidebar.slider("Số phiên muốn dự báo", min_value=1, max_value=30, value=10)
epochs = st.sidebar.slider("Số epoch tối đa", min_value=1, max_value=50, value=10)
train_button = st.sidebar.button("🚀 Chạy dự báo", type="primary", use_container_width=True)

st.sidebar.info(
    "Ví dụ mã Việt Nam: VNM.VN, FPT.VN, VIC.VN. "
    "Mã có thể thay đổi tùy khả năng hỗ trợ của Yahoo Finance."
)

if not ticker:
    st.warning("Vui lòng nhập mã cổ phiếu.")
    st.stop()

if start_date >= end_date:
    st.error("Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

with st.spinner(f"Đang tải dữ liệu {ticker}..."):
    stock_df = download_stock(ticker, start_date, end_date)

if stock_df.empty:
    st.error(
        "Không tải được dữ liệu. Hãy kiểm tra mã cổ phiếu, khoảng ngày hoặc kết nối Internet."
    )
    st.stop()

if len(stock_df) < 100:
    st.error("Dữ liệu có dưới 100 phiên giao dịch; hãy chọn khoảng thời gian dài hơn.")
    st.stop()

# -------------------- Tổng quan --------------------
latest_price = float(stock_df["Close"].iloc[-1])
previous_price = float(stock_df["Close"].iloc[-2])
change = latest_price - previous_price
change_pct = (change / previous_price * 100) if previous_price else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mã cổ phiếu", ticker)
c2.metric("Giá đóng cửa gần nhất", f"{latest_price:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
c3.metric("Số phiên dữ liệu", f"{len(stock_df):,}")
c4.metric("Ngày dữ liệu cuối", stock_df.index[-1].strftime("%d/%m/%Y"))

with st.expander("Xem dữ liệu lịch sử"):
    st.dataframe(stock_df.tail(20).style.format({"Close": "{:,.2f}"}), use_container_width=True)

history_fig, history_ax = plt.subplots(figsize=(12, 4))
history_ax.plot(stock_df.index, stock_df["Close"], label="Giá đóng cửa")
history_ax.set_title(f"Lịch sử giá đóng cửa - {ticker}")
history_ax.set_xlabel("Ngày")
history_ax.set_ylabel("Giá")
history_ax.grid(True, alpha=0.25)
history_ax.legend()
history_fig.tight_layout()
st.pyplot(history_fig)
plt.close(history_fig)

# Chỉ huấn luyện khi người dùng bấm nút
if not train_button:
    st.info("Chọn thông số ở thanh bên rồi bấm **Chạy dự báo** để huấn luyện mô hình LSTM.")
    st.stop()

# -------------------- Chuẩn bị dữ liệu --------------------
with st.spinner("Đang huấn luyện LSTM và tính toán kết quả..."):
    values = stock_df[["Close"]].values.astype("float32")
    train_size = int(0.8 * len(values))
    train_raw = values[:train_size]
    test_raw = values[train_size:]

    if len(train_raw) <= 60 or len(test_raw) < 2:
        st.error("Dữ liệu chưa đủ để chia tập huấn luyện và kiểm tra.")
        st.stop()

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train_all, y_train_all = create_sequences(train_scaled, window_size=60)
    val_size = max(1, int(0.1 * len(X_train_all)))
    if len(X_train_all) - val_size < 1:
        st.error("Không đủ dữ liệu để tạo tập xác thực.")
        st.stop()

    X_train = X_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_val = y_train_all[-val_size:]

    # Ghép 60 phiên cuối của tập train vào trước tập test để tạo chuỗi kiểm tra
    test_combined = np.vstack([train_scaled[-60:], test_scaled])
    X_test, y_test = create_sequences(test_combined, window_size=60)

    X_train = X_train.reshape((-1, 60, 1))
    X_val = X_val.reshape((-1, 60, 1))
    X_test = X_test.reshape((-1, 60, 1))

    model = build_model(window_size=60)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )

    test_pred_scaled = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred_scaled).ravel()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
    mae = float(mean_absolute_error(y_test_real, test_pred))
    r2 = float(r2_score(y_test_real, test_pred)) if len(y_test_real) > 1 else float("nan")

    # Dự báo lặp cho các phiên tiếp theo; ngày tương lai là ngày làm việc gần đúng,
    # không bao gồm thứ Bảy và Chủ nhật, nhưng không xử lý đầy đủ ngày nghỉ lễ.
    last_60 = np.vstack([train_scaled, test_scaled])[-60:].reshape(1, 60, 1)
    future_scaled = []
    for _ in range(forecast_days):
        next_value = model.predict(last_60, verbose=0)
        future_scaled.append(float(next_value[0, 0]))
        last_60 = np.concatenate([last_60[:, 1:, :], next_value.reshape(1, 1, 1)], axis=1)

    future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).ravel()
    future_dates = pd.bdate_range(
        start=stock_df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
    )
    future_df = pd.DataFrame(
        {"Ngày dự báo": future_dates.strftime("%d/%m/%Y"), "Giá dự báo": future_prices}
    )

# -------------------- Hiển thị kết quả --------------------
st.subheader("📊 Kết quả đánh giá mô hình")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:,.3f}")
m2.metric("MAE", f"{mae:,.3f}")
m3.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "Không xác định")

st.caption(
    "RMSE và MAE càng thấp thường càng tốt. R² có thể âm nếu mô hình dự báo kém hơn "
    "mốc so sánh đơn giản; các chỉ số này không đảm bảo hiệu quả trong tương lai."
)

left, right = st.columns(2)
with left:
    loss_fig, loss_ax = plt.subplots(figsize=(7, 4))
    loss_ax.plot(history.history["loss"], label="Train Loss")
    loss_ax.plot(history.history["val_loss"], label="Validation Loss")
    loss_ax.set_title("Loss: huấn luyện và xác thực")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Mean squared error")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend()
    loss_fig.tight_layout()
    st.pyplot(loss_fig)
    plt.close(loss_fig)

with right:
    compare_fig, compare_ax = plt.subplots(figsize=(7, 4))
    compare_ax.plot(np.arange(len(y_test_real)), y_test_real, label="Giá thật")
    compare_ax.plot(np.arange(len(test_pred)), test_pred, label="Dự báo trên tập test")
    compare_ax.set_title("Giá thật và giá dự báo trên tập kiểm tra")
    compare_ax.set_xlabel("Phiên kiểm tra")
    compare_ax.set_ylabel("Giá")
    compare_ax.grid(True, alpha=0.25)
    compare_ax.legend()
    compare_fig.tight_layout()
    st.pyplot(compare_fig)
    plt.close(compare_fig)

st.subheader(f"🔮 Dự báo {forecast_days} phiên giao dịch tiếp theo")
future_fig, future_ax = plt.subplots(figsize=(12, 5))
recent_count = min(60, len(stock_df))
recent = stock_df.iloc[-recent_count:]
future_ax.plot(recent.index, recent["Close"], marker=".", label="Giá lịch sử gần đây")
future_ax.plot(future_dates, future_prices, marker="o", linestyle="--", label="Giá dự báo")
future_ax.set_title(f"Dự báo giá đóng cửa - {ticker}")
future_ax.set_xlabel("Ngày")
future_ax.set_ylabel("Giá")
future_ax.grid(True, alpha=0.25)
future_ax.legend()
future_fig.tight_layout()
st.pyplot(future_fig)
plt.close(future_fig)

st.dataframe(
    future_df.style.format({"Giá dự báo": "{:,.2f}"}),
    use_container_width=True,
    hide_index=True,
)

csv = future_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Tải bảng dự báo CSV",
    data=csv,
    file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
    mime="text/csv",
)

st.warning(
    "Lưu ý phương pháp: mô hình dùng giá đóng cửa đơn biến và dự báo nhiều bước theo kiểu "
    "lặp, nên sai số có thể tích lũy. Ngày dự báo là ngày làm việc gần đúng, không phải lịch "
    "giao dịch chính thức của từng sàn. Không dùng kết quả này làm căn cứ duy nhất để đầu tư."
)

