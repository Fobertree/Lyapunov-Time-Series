import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def get_tkr_data(tkr, period='6mo', norm = False, csv_path = None):
    df =  yf.download(tkr, period=period)["Close"]

    if csv_path:
        df.to_csv(csv_path)

    if norm:
        scaler = MinMaxScaler()
        df.values = scaler.fit_transform(df.values.reshape(-1, 1)).flatten()
            
    return df