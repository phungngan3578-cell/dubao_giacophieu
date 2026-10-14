
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import random
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. CẤU HÌNH WEBSITE
# =========================
st.set_page_config(
    page_title="StockScope - Dự báo cổ phiếu",
    page_icon="📈",
    layout="wide"
)

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Giao diện
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #eaf0f7;
}
.hero {
    background: linear-gradient(120deg, #12243a, #28718a);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #e3edf5;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📈 StockScope</h1>
    <p>Hệ thống phân tích và dự báo giá cổ phiếu
    bằng mô hình học sâu LSTM</p>
</div>
""", unsafe_allow_html=True)

# =========================
# 2. HÀM XỬ LÝ DỮ LIỆU
# =========================
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        close_columns = [
            col for col in df.columns
            if col[0] == "Close"
        ]

        if not close_columns:
            return pd.DataFrame()

        close = df[close_columns[0]]
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()

        close = df["Close"]

    result = pd.DataFrame({
        "Close": pd.to_numeric(close, errors="coerce")
    })

    return result.dropna().sort_index()


def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def build_model():
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# =========================
# 3. THANH CẤU HÌNH
# =========================
st.sidebar.title("⚙️ Cấu hình")

ticker_input = st.sidebar.text_input(
    "Mã cổ phiếu",
    value="VNM"
).strip().upper()

start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    value=datetime.date(2018, 1, 1)
)

end_date = st.sidebar.date_input(
    "Ngày kết thúc",
    value=datetime.date.today()
)

n_days = st.sidebar.slider(
    "Số phiên dự báo",
    min_value=1,
    max_value=30,
    value=10
)

epochs = st.sidebar.slider(
    "Số vòng huấn luyện",
    min_value=5,
    max_value=30,
    value=10
)

btn_predict = st.sidebar.button(
    "🚀 Chạy dự báo",
    type="primary",
    use_container_width=True
)

ticker = ticker_input

if ticker and len(ticker) == 3 and "." not in ticker:
    ticker = ticker + ".VN"

if start_date >= end_date:
    st.error("Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

if not ticker:
    st.warning("Vui lòng nhập mã cổ phiếu.")
    st.stop()

# =========================
# 4. TẢI DỮ LIỆU
# =========================
with st.spinner("Đang tải dữ liệu cổ phiếu..."):
    df = get_stock_data(
        ticker,
        start_date,
        end_date
    )

if df.empty:
    st.error(
        "Không tìm thấy dữ liệu. "
        "Hãy kiểm tra mã cổ phiếu hoặc kết nối Internet."
    )
    st.stop()

if len(df) < 100:
    st.error("Dữ liệu chưa đủ 100 phiên giao dịch.")
    st.stop()

data = df[["Close"]].values.astype(float)

latest_price = float(data[-1, 0])
previous_price = float(data[-2, 0])

daily_change = latest_price - previous_price

daily_pct = (
    daily_change / previous_price * 100
    if previous_price != 0 else 0
)

# =========================
# 5. TỔNG QUAN THỊ TRƯỜNG
# =========================
st.subheader("📊 Tổng quan cổ phiếu")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Mã cổ phiếu", ticker)

c2.metric(
    "Giá đóng cửa gần nhất",
    f"{latest_price:,.2f}",
    f"{daily_change:+,.2f}"
)

c3.metric(
    "Biến động phiên gần nhất",
    f"{daily_pct:+.2f}%"
)

c4.metric(
    "Số phiên dữ liệu",
    f"{len(df):,}"
)

# Biểu đồ lịch sử
history_fig = go.Figure()

history_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    mode="lines",
    name="Giá đóng cửa",
    line=dict(color="#2878b5", width=2)
))

history_fig.update_layout(
    title=f"Lịch sử giá cổ phiếu {ticker}",
    xaxis_title="Ngày",
    yaxis_title="Giá đóng cửa",
    template="plotly_white",
    height=400,
    hovermode="x unified"
)

st.plotly_chart(
    history_fig,
    use_container_width=True
)

with st.expander("Xem dữ liệu lịch sử"):
    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

# =========================
# 6. HUẤN LUYỆN MÔ HÌNH
# =========================
if not btn_predict:
    st.info(
        "Chọn thông số ở thanh bên rồi nhấn "
        "**Chạy dự báo** để bắt đầu."
    )
    st.stop()

with st.spinner("Đang huấn luyện mô hình LSTM..."):

    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(
        train_scaled,
        window_size=60
    )

    # Ghép 60 phiên trước tập test
    test_combined = np.vstack([
        train_scaled[-60:],
        test_scaled
    ])

    X_test, y_test = create_sequences(
        test_combined,
        window_size=60
    )

    # Chia tập validation theo thứ tự thời gian
    val_size = max(1, int(len(X_train) * 0.1))

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    X_fit = X_train[:-val_size]
    y_fit = y_train[:-val_size]

    if len(X_fit) == 0:
        st.error("Không đủ dữ liệu để huấn luyện.")
        st.stop()

    model = build_model()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Dự báo tập kiểm tra
    test_pred = model.predict(
        X_test,
        verbose=0
    )

    test_pred = scaler.inverse_transform(test_pred)

    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    # Chỉ số đánh giá
    rmse = np.sqrt(
        mean_squared_error(y_test_real, test_pred)
    )

    mae = mean_absolute_error(
        y_test_real,
        test_pred
    )

    r2 = r2_score(
        y_test_real,
        test_pred
    )

    # =========================
    # 7. DỰ BÁO TƯƠNG LAI
    # =========================
    last_60 = np.vstack([
        train_scaled,
        test_scaled
    ])[-60:]

    last_60 = last_60.reshape(1, 60, 1)

    future_predictions = []

    for _ in range(n_days):

        next_pred = model.predict(
            last_60,
            verbose=0
        )

        future_predictions.append(
            next_pred[0, 0]
        )

        last_60 = np.concatenate([
            last_60[:, 1:, :],
            next_pred.reshape(1, 1, 1)
        ], axis=1)

    future_predictions = np.array(
        future_predictions
    ).reshape(-1, 1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    ).flatten()

# =========================
# 8. TÓM TẮT DỰ BÁO
# =========================
final_price = float(future_predictions[-1])

price_diff = final_price - latest_price

pct_change = (
    price_diff / latest_price * 100
    if latest_price != 0 else 0
)

st.markdown("---")
st.subheader("🎯 Kết quả dự báo")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Giá hiện tại",
    f"{latest_price:,.2f}"
)

m2.metric(
    f"Giá dự báo phiên {n_days}",
    f"{final_price:,.2f}"
)

m3.metric(
    "Chênh lệch ước tính",
    f"{price_diff:+,.2f}"
)

m4.metric(
    "Biến động ước tính",
    f"{pct_change:+.2f}%"
)

# =========================
# 9. BẢNG DỰ BÁO
# =========================
st.subheader("📋 Bảng dự báo chi tiết")

last_date = pd.Timestamp(df.index[-1])

future_dates = pd.bdate_range(
    start=last_date + pd.Timedelta(days=1),
    periods=n_days
)

forecast_df = pd.DataFrame({
    "Ngày dự báo": future_dates.strftime("%d/%m/%Y"),
    "Giá dự báo": future_predictions,
    "Thay đổi (%)": [
        (p - latest_price) / latest_price * 100
        if latest_price else 0
        for p in future_predictions
    ]
})

st.dataframe(
    forecast_df.style.format({
        "Giá dự báo": "{:,.2f}",
        "Thay đổi (%)": "{:+.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "⬇️ Tải bảng dự báo CSV",
    data=forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
    mime="text/csv"
)

# =========================
# 10. BIỂU ĐỒ PHÂN TÍCH
# =========================
st.markdown("---")
st.subheader("📉 Phân tích mô hình")

tab1, tab2, tab3 = st.tabs([
    "Giá thực tế và dự báo",
    "Quá trình huấn luyện",
    "Đánh giá mô hình"
])

with tab1:

    fig = go.Figure()

    test_dates = df.index[train_size:]

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test_real.flatten(),
        mode="lines",
        name="Giá thực tế",
        line=dict(color="#2878b5", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_pred.flatten(),
        mode="lines",
        name="Giá dự báo trên tập test",
        line=dict(
            color="#f59e0b",
            width=2,
            dash="dash"
        )
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode="lines+markers",
        name="Dự báo tương lai",
        line=dict(color="#16a34a", width=3)
    ))

    fig.update_layout(
        title=f"So sánh giá thực tế và dự báo - {ticker}",
        xaxis_title="Ngày",
        yaxis_title="Giá cổ phiếu",
        template="plotly_white",
        height=480,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    loss_fig = go.Figure()

    loss_fig.add_trace(go.Scatter(
        y=history.history["loss"],
        mode="lines+markers",
        name="Train Loss"
    ))

    loss_fig.add_trace(go.Scatter(
        y=history.history["val_loss"],
        mode="lines+markers",
        name="Validation Loss"
    ))

    loss_fig.update_layout(
        title="Loss qua các vòng huấn luyện",
        xaxis_title="Epoch",
        yaxis_title="Mean Squared Error",
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        loss_fig,
        use_container_width=True
    )

with tab3:

    a, b, c = st.columns(3)

    a.metric("RMSE", f"{rmse:,.4f}")
    b.metric("MAE", f"{mae:,.4f}")
    c.metric("R² Score", f"{r2:.4f}")

    st.write("""
    - **RMSE:** sai số bình phương trung bình đã lấy căn.
    - **MAE:** sai số tuyệt đối trung bình.
    - **R²:** chỉ số phản ánh mức độ phù hợp của mô hình.
    
    RMSE và MAE càng thấp thường càng tốt. R² có thể âm
    nếu mô hình dự báo kém hơn một mốc so sánh đơn giản.
    """)

st.markdown("---")

st.caption("""
StockScope phục vụ mục đích học tập và nghiên cứu.
Dự báo LSTM có thể sai lệch so với diễn biến thực tế,
đặc biệt khi dự báo nhiều phiên liên tiếp.
Không xem kết quả là cam kết lợi nhuận hoặc khuyến nghị đầu tư.
""")


