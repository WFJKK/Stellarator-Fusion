import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, train_dataset, test_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_test_loss = float('inf')
    best_model_path = "best_model.pth"

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: New best model saved (Test Loss: {test_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
