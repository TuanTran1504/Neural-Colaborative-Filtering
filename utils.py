import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dims, num_days=7):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Embedding for categorical features
        self.day_embedding = nn.Embedding(num_days, 4)  # Example: Embedding size of 4 for day of the week
  # Example: Embedding size of 4 for hour of the day

        # Fully connected layers
        input_dim = embedding_dim * 2 + 1 + 4  # User, item, timestamp, day embedding, hour embedding
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = dim
        self.mlp = nn.Sequential(*layers)

        self.final_layer = nn.Linear(input_dim, 1)

    def forward(self, user_ids, item_ids, timestamps, day_of_week):
        # Get embeddings
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        # Get day and hour embeddings
        day_embed = self.day_embedding(day_of_week)

        # Concatenate all features
        x = torch.cat([user_embed, item_embed, timestamps.unsqueeze(1), day_embed], dim=-1)

        # Pass through fully connected layers
        x = self.mlp(x)
        rating_pred = torch.sigmoid(self.final_layer(x)) * 5  # Scale to [0, 5]
        return rating_pred


def evaluate_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch["user"].to(device)
            movie_ids = batch["movie"].to(device)
            ratings = batch["rating"].to(device)
            timestamps = batch["timestamp"].to(device)
            day_of_week = batch["dayofweek"].to(device)

            predictions = model(user_ids, movie_ids, timestamps, day_of_week)
            all_preds.extend(predictions.squeeze().tolist())
            all_targets.extend(ratings.tolist())

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return rmse


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim, mlp_dims):
        super(NeuMF, self).__init__()

        # GMF component
        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)

        # MLP component
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dims[0])
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dims[0])

        # MLP layers
        mlp_layers = []
        input_dim = mf_dim * 2
        dropout_rate = 0.5
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout_rate))
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final NeuMF layer
        self.final_layer = nn.Linear(mf_dim + mlp_dims[-1], 1)

    def forward(self, user_ids, item_ids):
        # GMF forward pass
        mf_user = self.mf_user_embedding(user_ids)
        mf_item = self.mf_item_embedding(item_ids)
        mf_vector = mf_user * mf_item  # Element-wise product
        # print(f"mf_user shape: {mf_user.shape}")
        # print(f"mf_item shape: {mf_item.shape}")
        # print(f"mf_vector shape: {mf_vector.shape}")
        # MLP forward pass
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_vector = torch.cat([mlp_user, mlp_item], dim=-1)
        # print(f"mlp_user shape: {mlp_user.shape}")
        # print(f"mlp_item shape: {mlp_item.shape}")
        # print(f"mlp_vector shape: {mlp_vector.shape}")
        try:
            mlp_output = self.mlp(mlp_vector)  # Shape: (batch_size, mlp_dims[-1])
            # Debugging - print shapes
            # print(f"mlp_output shape: {mlp_output.shape}")
        except Exception as e:
            # print(f"Error in MLP forward pass: {e}")
            # print(f"mlp_vector shape was: {mlp_vector.shape}")
            raise e

        # Combine GMF and MLP components
        combined = torch.cat([mf_vector, mlp_output], dim=-1)
        # print(f"combined shape: {combined.shape}")
        prediction = torch.sigmoid(self.final_layer(combined)) * 5  # Assuming rating scale 1-5

        return prediction


def test_model(model, train_loader, test_loader, optimizer, loss_function, epochs=10, device='cpu'):
    model.to(device)
    model.train()

    # Initialize scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    rmse_history = []  # To store RMSE for each epoch
    loss_history = []  # To store average training loss for each epoch
    # To store NDCG@k for each epoch

    for epoch in range(epochs):
        total_loss = 0
        model.train()  # Set the model to training mode
        for batch in train_loader:
            user_ids = batch['user'].to(device)
            movie_ids = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(user_ids, movie_ids).squeeze()

            # Calculate loss
            loss = loss_function(predictions, ratings)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss for each batch
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on the test set to calculate RMSE, HR@k, and NDCG@k
        rmse= evaluate_test_model(model, test_loader, device=device)
        rmse_history.append(rmse)


        print(f"Test RMSE after Epoch {epoch + 1}: {rmse:.4f}")

        # Step the scheduler with RMSE as the monitored metric
        scheduler.step(rmse)
        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch + 1}: Learning Rate = {current_lr}")
    # Plot RMSE, HR@k, NDCG@k, and Loss over epochs
    plt.figure(figsize=(12, 6))

    # Plot Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)

    # Plot RMSE
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs + 1), rmse_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Test RMSE")
    plt.title("Test RMSE Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Train the model
def evaluate_test_model(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []
    num_users = 0

    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch['user'].to(device)
            movie_ids = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            # Forward pass
            predictions = model(user_ids, movie_ids).squeeze()

            # Append predictions and targets for RMSE calculation
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return rmse
