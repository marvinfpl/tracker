import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import model
from sklearn.model_selection import train_test_split
import os

# import Dataset images

X: np.ndarray
y: np.ndarray
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), os.getenv('learning_rate'))

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp)

for epoch in os.getenv('epochs'):
    ids = np.random.choice(X, os.getenv('batch_size'))
    X_batch, y_batch = X_train[ids], y_train[ids]

    y_pred = model(X_batch)
    loss = loss_fn(y_pred, y_batch)

    if epoch % os.getenv('print_rate') == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()} ')

    model.zero_grad()
    loss.backward()
    optimizer.step()
    