from archive.pretrain import GMF
from archive.pretrain2 import MLP
from preprocessing import *
from utils import *
import matplotlib.pyplot as plt

train_loader, test_loader, df = preprocess("training_data","processed_data_test.csv", batch_size=512)

num_users = df.CustomerID.nunique()
num_items = df.MovieID.nunique()
print(num_users, num_items)

embedding_dim = 128
hidden_dims = [64, 32]
# model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim, hidden_dims)

model = NeuMF(num_users=num_users, num_items=num_items, mf_dim=128, mlp_dims=[256, 64,32])
# Load pre-trained GMF and MLP weights
gmf_model = GMF(num_users, num_items, embedding_dim=128)
mlp_model = MLP(num_users, num_items, embedding_dim=128, hidden_layers=[64,32])
print("Pretrained model loaded")
gmf_model.load_state_dict(torch.load('gmf_weights.pth'))
mlp_model.load_state_dict(torch.load('mlp_weights.pth'))
print("Pre-trained weights loaded successfully.")
# Set the weights in NeuMF using pre-trained values
model.mf_user_embedding.weight.data = gmf_model.user_embedding.weight.data
model.mf_item_embedding.weight.data = gmf_model.item_embedding.weight.data
model.mlp_user_embedding.weight.data = mlp_model.user_embedding.weight.data
model.mlp_item_embedding.weight.data = mlp_model.item_embedding.weight.data
print("Pre-trained weights have been set in NeuMF model.")
embedding_params = []
fc_params = []

for name, param in model.named_parameters():
    if "embedding" in name:  # Identify if the parameter belongs to embeddings
        embedding_params.append(param)
    else:
        fc_params.append(param)


optimizer = optim.SGD([
    {"params": embedding_params, "weight_decay": 0.0},  # No regularization for embeddings
    {"params": fc_params, "weight_decay": 1e-4}         # L2 regularization for fully connected layers
], lr=0.01, momentum=0.9)
# Define optimizer and loss function

# loss_function = nn.SmoothL1Loss()
loss_function = nn.MSELoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_model(model, train_loader, optimizer, loss_function, epochs=50, device=device)
print("Begin to train")
test_model(model, train_loader, test_loader, optimizer, loss_function, epochs=50, device=device)