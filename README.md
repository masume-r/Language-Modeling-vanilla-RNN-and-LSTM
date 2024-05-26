comparison of the language generation performances of a Vanilla RNN and an LSTM model
epochs = 20 

Epoch 1/20, RNN Training Loss: 2.1305, RNN Validation Loss: 1.8888
Epoch 1/20, LSTM Training Loss: 2.0558, LSTM Validation Loss: 1.6438
Epoch 2/20, RNN Training Loss: 1.9272, RNN Validation Loss: 1.8011
Epoch 2/20, LSTM Training Loss: 1.6541, LSTM Validation Loss: 1.4712
Epoch 3/20, RNN Training Loss: 1.8619, RNN Validation Loss: 1.7414
Epoch 3/20, LSTM Training Loss: 1.5421, LSTM Validation Loss: 1.3845
Epoch 4/20, RNN Training Loss: 1.8199, RNN Validation Loss: 1.6991
Epoch 4/20, LSTM Training Loss: 1.4764, LSTM Validation Loss: 1.3270
Epoch 5/20, RNN Training Loss: 1.7900, RNN Validation Loss: 1.6805
Epoch 5/20, LSTM Training Loss: 1.4307, LSTM Validation Loss: 1.2862
Epoch 6/20, RNN Training Loss: 1.7676, RNN Validation Loss: 1.6577
Epoch 6/20, LSTM Training Loss: 1.3974, LSTM Validation Loss: 1.2547
Epoch 7/20, RNN Training Loss: 1.7496, RNN Validation Loss: 1.6329
Epoch 7/20, LSTM Training Loss: 1.3720, LSTM Validation Loss: 1.2325
.
.
.
Epoch 17/20, RNN Training Loss: 1.6736, RNN Validation Loss: 1.5576
Epoch 17/20, LSTM Training Loss: 1.2683, LSTM Validation Loss: 1.1314
Epoch 18/20, RNN Training Loss: 1.6701, RNN Validation Loss: 1.5524
Epoch 18/20, LSTM Training Loss: 1.2634, LSTM Validation Loss: 1.1209
Epoch 19/20, RNN Training Loss: 1.6669, RNN Validation Loss: 1.5477
Epoch 19/20, LSTM Training Loss: 1.2589, LSTM Validation Loss: 1.1190
Epoch 20/20, RNN Training Loss: 1.6634, RNN Validation Loss: 1.5448
Epoch 20/20, LSTM Training Loss: 1.2551, LSTM Validation Loss: 1.1164




 LSTM model consistently outperforms the Vanilla RNN model in terms of both training and validation losses. The LSTM model shows a more substantial reduction in loss values over the epochs
 * the LSTM's training loss decreases more rapidly than the RNN's training loss. By the end of the 20 epochs, the LSTM's training loss is significantly lower.
 * Similarly, the validation loss for the LSTM model decreases more consistently compared to the RNN model


plot the training and validation losses for Vanilla RNN and an LSTM model : 

![rnn](https://github.com/masume-r/model.py/assets/167098630/911c9df4-3fc0-4467-bcb0-4a5d39fd5ab0)
![lstm](https://github.com/masume-r/model.py/assets/167098630/f3db64f2-8852-44e7-b8c9-48e711a79e45)

 
Temperature parameter: 

Low Temperature (T < 1):
tends to be more repetitive and predictable and lacks diversity

Moderate Temperature (T = 1):
balance between diversity and coherence

High Temperature (T > 1):
more creative and diverse.


