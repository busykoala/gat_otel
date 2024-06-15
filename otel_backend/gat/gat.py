import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(
        self,
        optimizer,
        num_features,
        num_classes,
        weight_decay=1e-3,
        dropout=0.7,
        hidden_dim=16,
        lr=0.005,
        patience=3,
    ):
        super(GAT, self).__init__()
        self.patience = patience
        self.conv1 = GATConv(num_features, hidden_dim, heads=8, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 8)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 8)
        self.conv3 = GATConv(
            hidden_dim * 8, num_classes, heads=1, concat=True, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=self.patience, verbose=True
        )

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

    def train_step(self, data):
        self.train()
        self.optimizer.zero_grad()
        x, edge_index, y = data.x, data.edge_index, data.y

        out = self(x, edge_index)
        loss = F.nll_loss(out, y)
        loss.backward()
        self.optimizer.step()

        _, preds = out.max(dim=1)
        correct = preds.eq(y).sum().item()
        accuracy = correct / y.size(0)

        return loss.item(), accuracy

    def train_model(self, data):
        best_val_loss = float("inf")
        patience_counter = 0
        iteration = 0

        while True:
            iteration += 1
            train_loss, train_accuracy = self.train_step(data)

            print(
                f"Iteration {iteration}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
            )

            # Adjust learning rate based on training loss
            self.scheduler.step(train_loss)
            print(
                f'Iteration {iteration}, Current Learning Rate: {self.scheduler.optimizer.param_groups[0]["lr"]}'
            )

            # Early stopping check to prevent overfitting
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                patience_counter = 0
                torch.save(self.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping")
                    break

        self.load_state_dict(torch.load("best_model.pth"))

    def test_model(self, data):
        self.eval()
        x, edge_index = data.x, data.edge_index
        out = self(x, edge_index)
        _, pred = out.max(dim=1)
        return pred
