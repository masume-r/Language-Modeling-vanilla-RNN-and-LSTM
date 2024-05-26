import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import dataset
from model import CharRNN, CharLSTM


def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0

    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        inputs = one_hot_encode(inputs, model.fc.out_features).to(device)
        hidden = model.init_hidden(inputs.size(0))
        hidden = tuple([h.to(device) for h in hidden]) if isinstance(hidden, tuple) else hidden.to(device)

        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

    trn_loss /= len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = one_hot_encode(inputs, model.fc.out_features).to(device)
            hidden = model.init_hidden(inputs.size(0))
            hidden = tuple([h.to(device) for h in hidden]) if isinstance(hidden, tuple) else hidden.to(device)

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0

    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):
            hidden = tuple([h.to(device) for h in hidden])
        else:
            hidden = hidden.to(device)

        inputs = nn.functional.one_hot(inputs, num_classes=model.fc.out_features).float()
        outputs, hidden = model(inputs, hidden)

        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

    trn_loss /= len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple([h.to(device) for h in hidden])
            else:
                hidden = hidden.to(device)

            inputs = nn.functional.one_hot(inputs, num_classes=model.fc.out_features).float()
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs, targets.view(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main():
    input_file = "shakespeare_train.txt"
    batch_size = 64
    num_epochs = 3
    learning_rate = 0.001
    validation_split = 0.2
    weight_decay = 1e-5 

    dataset = Shakespeare(input_file)
    vocab_size = len(dataset.char_to_idx)

    train_size = int(len(dataset) * (1 - validation_split))
    val_size = len(dataset) - train_size
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    trn_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=trn_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    rnn_model = CharRNN(vocab_size, 256, vocab_size, dropout=0.5)  
    lstm_model = CharLSTM(vocab_size, 256, vocab_size, dropout=0.5)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_model.to(device)
    lstm_model.to(device)

    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    rnn_train_losses = []
    rnn_val_losses = []
    lstm_train_losses = []
    lstm_val_losses = []

    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    for epoch in range(num_epochs):
      
        rnn_trn_loss = train(rnn_model, trn_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        rnn_train_losses.append(rnn_trn_loss)
        rnn_val_losses.append(rnn_val_loss)

        lstm_trn_loss = train(lstm_model, trn_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)
        lstm_train_losses.append(lstm_trn_loss)
        lstm_val_losses.append(lstm_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, RNN Training Loss: {rnn_trn_loss:.4f}, RNN Validation Loss: {rnn_val_loss:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs}, LSTM Training Loss: {lstm_trn_loss:.4f}, LSTM Validation Loss: {lstm_val_loss:.4f}')

        early_stopping(rnn_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(rnn_train_losses) + 1), rnn_train_losses, label='RNN Training Loss')
    plt.plot(range(1, len(rnn_val_losses) + 1), rnn_val_losses, label='RNN Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training & Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(lstm_train_losses) + 1), lstm_train_losses, label='LSTM Training Loss')
    plt.plot(range(1, len(lstm_val_losses) + 1), lstm_val_losses, label='LSTM Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training & Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
