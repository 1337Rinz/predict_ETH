import yfinance as yf
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dcc, html
from datetime import date, timedelta
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
import plotly.graph_objects as go
import os



# os.chdir('D:/pythonProject/CryptoPredict/app')



# Xác định ký hiệu mã cổ phiếu và phạm vi ngày
stock_symbol = "ETH-USD"

# Lấy ngày hiện tại
end_date = date.today().strftime('%Y-%m-%d')

# Lấy ngày 60 ngày trước
start_date = (date.today() - timedelta(days=61)).strftime('%Y-%m-%d')

# Lấy dữ liệu lịch sử giá Ethereum từ 60 ngày trước đến ngày hiện tại
ETH = yf.download(stock_symbol, start=start_date, end=end_date)
ETH_realtime = ETH[['Close']]

# load scaler
scaler = joblib.load('models/min_max_scaler.pkl')

# Chuẩn hóa dữ liệu
data_normalized = scaler.transform(ETH_realtime.values)

# Chuẩn bị dữ liệu cho model
X_test = []
X_test.append(data_normalized[-60:, 0])
X_test = np.array(X_test)

# Đường dẫn tới tệp model
model_path = "models/best_LstmModel_2024.h5"

# Tải model từ tệp
model = load_model(model_path)

# Dự đoán giá cho 3 ngày tiếp theo
predicted_prices = model.predict(X_test)

# Sử dụng bộ chuẩn hóa đã được lưu
# scaler = joblib.load('models/min_max_scaler.pkl')

# Chuẩn hóa lại predicted_prices
predicted_prices_scaled = scaler.inverse_transform(predicted_prices.reshape(-1, 1)).ravel()
print("predicted_prices: ",predicted_prices_scaled)

# Tạo DataFrame từ predicted_prices_scaled
df_predicted_prices = pd.DataFrame({'Prediction': predicted_prices_scaled})

# Ngày cuối cùng trong ETH_realtime
last_date = ETH_realtime.index[-1]

# Create a list of dates for the predicted prices
dates = [last_date + timedelta(days=i) for i in range(1, 4)]

# Convert the list to a pandas datetime index
dates = pd.DatetimeIndex(dates)


# Tạo DataFrame từ dữ liệu dự đoán và ngày tương ứng
df_predicted_prices['Date'] = dates
print("predict table: \n",df_predicted_prices)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1('Dự đoán giá Ethereum (ETH-USD)'),
dcc.Graph(id='graph'),
    dcc.Slider(
        id='date-slider',
        min=0,
        max=len(ETH_realtime)-1,
        value=len(ETH_realtime)-1,
        marks={i: ETH_realtime.index[i] for i in range(0, len(ETH_realtime), 10)},
        step=None
    )
])

@app.callback(
    Output('graph', 'figure'),
    [Input('date-slider', 'value')]
)
def update_graph(selected_date):
    # Lấy dữ liệu ETH_realtime từ ngày được chọn
    ETH_realtime_selected = ETH_realtime.iloc[:selected_date+1]

    # Tạo biểu đồ
    fig = go.Figure()

    # Thêm đường
    fig.add_trace(go.Scatter(x=ETH_realtime_selected.index, y=ETH_realtime_selected['Close'], name='Giá thực tế'))
    fig.add_trace(go.Scatter(x=df_predicted_prices['Date'], y=df_predicted_prices['Prediction'], name='Dự đoán'))

    # Cài đặt trục
    fig.update_layout(
        title='Giá Ethereum (ETH-USD)',
        xaxis_title='Ngày',
        yaxis_title='Giá'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)