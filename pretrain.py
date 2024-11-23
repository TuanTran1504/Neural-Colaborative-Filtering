import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
from preprocessing import *

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        gmf_vector = user_embed * item_embed  # Element-wise product
        prediction = torch.sigmoid(gmf_vector.sum(dim=1)) * 5  # Scale to [0, 5]
        return prediction

if __name__ == "__main__":
    train_loader, df = preprocess("training_data","processed_data.csv", batch_size=512, pretrain=True)

    # Initialize GMF model
    num_users = df.CustomerID.nunique()
    num_items = df.MovieID.nunique()
    embedding_dim = 128

    gmf_model = GMF(num_users, num_items, embedding_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gmf_model = gmf_model.to(device)

    # Define loss and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(gmf_model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train GMF model
    gmf_model.train()
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
            predictions = gmf_model(user_ids, movie_ids).squeeze()

            # Compute the loss
            loss = loss_function(predictions, ratings)

            # Backward pass and update
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save GMF model weights
    torch.save(gmf_model.state_dict(), 'gmf_weights_2.pth')