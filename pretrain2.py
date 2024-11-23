import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
from preprocessing import *

class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.hidden_layers = nn.Sequential()
        input_dim = embedding_dim * 2

        for i, dim in enumerate(hidden_layers):
            self.hidden_layers.add_module(f"linear_{i}", nn.Linear(input_dim, dim))
            self.hidden_layers.add_module(f"relu_{i}", nn.ReLU())
            input_dim = dim

        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        x = torch.cat([user_embed, item_embed], dim=-1)  # Concatenate embeddings
        x = self.hidden_layers(x)
        prediction = torch.sigmoid(self.output_layer(x)).squeeze() * 5  # Scale to [0, 5]
        return prediction


if __name__ == "__main__":
    train_loader, df = preprocess("training_data","processed_data_test.csv", batch_size=512, pretrain=True)

    # Initialize GMF model
    num_users = df.CustomerID.nunique()
    num_items = df.MovieID.nunique()
    mlp_model = MLP(num_users, num_items, embedding_dim=128, hidden_layers=[64, 32])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model.to(device)
    loss_function = nn.MSELoss()

    # Define loss and optimizer
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train MLP model
    # mlp_model.train()
    epochs=50
    for epoch in range(epochs):  # Training for 10 epochs as an example
        total_loss = 0
        for batch in train_loader:
            user_ids = batch['user'].to(device)
            movie_ids = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = mlp_model(user_ids, movie_ids).squeeze()

            # Compute the loss
            loss = loss_function(predictions, ratings)

            # Backward pass and update
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save MLP model weights
    torch.save(mlp_model.state_dict(), 'mlp_weights.pth')