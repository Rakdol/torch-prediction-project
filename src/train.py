import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=300,
    patience=50,
    min_delta=1e-4,
    lr=1e-3,
    weight_decay=1e-4,
    scheduler_patience=10,
    factor=0.5,
    save_path="best_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=scheduler_patience, factor=factor, verbose=True
    )

    best_model = deepcopy(model.state_dict())
    best_loss = float("inf")
    wait = 0

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_loss = criterion(model(X_val), y_val)
                val_losses.append(val_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(
            f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        scheduler.step(val_loss)

        if val_loss + min_delta < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model.state_dict())
            wait = 0
            torch.save(best_model, save_path)
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"⏹ Early stopping at epoch {epoch}. Best Val Loss: {best_loss:.4f}"
                )
                break

    model.load_state_dict(best_model)
    model.to("cpu")

    return model, train_loss_history, val_loss_history
