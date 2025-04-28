import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据集类构造
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class LSTM_Transformer(nn.Module):
    def __init__(self, input_size=1, lstm_hidden_size=64, lstm_layers=2,
                 trans_layers=2, trans_heads=4, trans_ffn_hidden=128, dropout=0.1, output_size=1, output_window=10, activation='relu'):
        super(LSTM_Transformer, self).__init__()
        if lstm_layers == 1:
            dropout = 0
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.output_window = output_window

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_layers, batch_first=True, dropout=dropout)

        self.pos_encoder = PositionalEncoding(d_model=lstm_hidden_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_size, nhead=trans_heads,
                                                   dim_feedforward=trans_ffn_hidden, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()  

        self.fc = nn.Linear(lstm_hidden_size, output_size * output_window)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, lstm_hidden_size)

        trans_in = self.pos_encoder(lstm_out)

        trans_out = self.transformer_encoder(trans_in)  # (batch, seq_len, lstm_hidden_size)

        final_feature = trans_out[:, -1, :]  # (batch, lstm_hidden_size)

        final_feature = self.activation(final_feature)

        output = self.fc(final_feature).view(batch_size, self.output_window, -1)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class WeightedMSELoss(nn.Module):
    def __init__(self, weights, device):
        super(WeightedMSELoss, self).__init__()
        if torch.isnan(torch.tensor(weights)).any() or torch.isinf(torch.tensor(weights)).any():
            raise ValueError("Weights contain NaN or Inf values. Please check the weight calculation.")
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        weighted_squared_error = self.weights * squared_error
        return torch.mean(weighted_squared_error)
    
def build_and_train_Transformer_bys_weighting(X_train, y_train, batch_size, output_window, epochs=1000, input_size=1,
                                     lstm_hidden_size=64, lstm_layers=2, trans_layers=2, trans_heads=4,
                                     trans_ffn_hidden=128, dropout=0.1, output_size=1,
                                     learning_rate=0.001, lr_decay_factor=0.1, activation='relu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTM_Transformer(input_size=input_size, lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers,
                             trans_layers=trans_layers, trans_heads=trans_heads, trans_ffn_hidden=trans_ffn_hidden,
                             dropout=dropout, output_size=output_size, output_window=output_window,
                             activation=activation).to(device)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    criterion = nn.MSELoss().to(device)
    train_losses = []

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_factor)

    num_intervals = (output_window + 5) // 6
    start_weight = 0.8
    decay_rate = 0.2
    weights_per_6h = [max(0.1, start_weight - i * decay_rate) for i in range(num_intervals)]

    interval_length = 6*4

    weights = []
    for weight in weights_per_6h:
        if len(weights) + interval_length <= output_window:
            weights.extend([weight] * interval_length)
        else:
            remaining_length = output_window - len(weights)
            weights.extend([weight] * remaining_length)
            break

    weights = weights[:output_window]

    criterion = WeightedMSELoss(weights, device).to(device)
    train_losses = []

    num_epochs = epochs
    model.train()
    print("Training...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss).any():
                print("Loss is NaN. Stopping training.")
                return model, train_losses
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.8f}")

        scheduler.step()

    return model, train_losses