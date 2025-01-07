import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load Data from SQL Database
def load_data_from_sql(ticker, start_date, end_date):
    # Connect to your SQL database
    conn = sqlite3.connect('stock_data.db')  # connect to database
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, closing_price, average_volume, market_cap, mma_30, mma_90
        FROM stocks 
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date ASC
    ''', (ticker, start_date, end_date))
    data = cursor.fetchall()
    conn.close()

    if not data:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return None

    # Convert to a DataFrame for easier processing
    df = pd.DataFrame(data, columns=['date', 'closing_price', 'average_volume', 'market_cap', 'mma_30', 'mma_90'])
    return df

stocks = ['AAPL']  # Example tickers
start_date = '2022-08-01'
end_date = '2024-12-13'
data = []
for ticker in stocks:
    stock_data = load_data_from_sql(ticker, start_date, end_date)
    if stock_data is not None:
        data.append(stock_data)

# Concatenate all stock data into a single DataFrame
data = pd.concat(data, ignore_index=True) if data else None

# Step 2: Preprocess Data
def preprocess_data(data):
    # Assume the table has columns: 'closing_price', 'average_volume', 'market_cap', 'mma_30', 'mma_90'
    features = data[['closing_price', 'average_volume', 'market_cap', 'mma_30', 'mma_90']].values
    target = data['closing_price'].values  # Use closing_price as the target
    return features, target

if data is not None:
    features, target = preprocess_data(data)
else:
    print("No data to preprocess.")

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
target = scaler.fit_transform(target.reshape(-1, 1)).flatten()

# Prepare sequences for LSTM
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Use 10 days of data to predict the next day
X, y = create_sequences(features, target, seq_length)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Define Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 4: Define the LSTM Model
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # Take the hidden state from the last LSTM layer
        out = self.fc(hn[-1])  # Pass through fully connected layer
        return out

input_size = 5
hidden_size = 128
output_size = 1
num_layers = 4

model = StockPriceLSTM(input_size, hidden_size, output_size, num_layers).to(device)

# Step 5: Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the Model
epochs = 100
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            test_loss += loss.item() * X_batch.size(0)

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Step 7: Save Model Weights
torch.save(model.state_dict(), "stock_price_lstm.pth")

# Step 8: Plot Predicted vs Actual Prices
model.eval()
predicted, actual = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch).squeeze()
        predicted.extend(output.cpu().numpy())
        actual.extend(y_batch.cpu().numpy())

# Rescale back to original prices
predicted = scaler.inverse_transform(np.array(predicted).reshape(-1, 1)).flatten()
actual = scaler.inverse_transform(np.array(actual).reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 6))
plt.plot(actual, label="Actual Price")
plt.plot(predicted, label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Predicted vs Actual Stock Prices")
plt.legend()
plt.show()