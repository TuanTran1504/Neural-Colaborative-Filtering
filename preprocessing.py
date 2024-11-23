import os
import pandas as pd
import torch
from click.core import batch
from torch.utils.data import Dataset
from sklearn import model_selection
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def preprocess(train_data_dir: str, file: str, batch_size: int, pretrain=False):
    # Load and concatenate all CSV files
    all_dataframes = []
    file_path = os.path.join(train_data_dir, file)

    # Check if the file exists and is a CSV
    if os.path.isfile(file_path) and file.endswith(".csv"):
        df = pd.read_csv(file_path)
        all_dataframes.append(df)
    else:
        raise FileNotFoundError(f"File {file} not found in directory {train_data_dir} or is not a CSV file.")

    class NetflixDataset(Dataset):
        def __init__(self, users, movies, ratings, timestamps, dayofweek):
            # Convert inputs to NumPy arrays for efficient indexing
            self.users = users
            self.movies = movies
            self.ratings = ratings
            self.timestamps = timestamps
            self.dayofweek = dayofweek


        def __len__(self):
            return len(self.users)

        def __getitem__(self, item):
            user = self.users[item]
            movie = self.movies[item]
            rating = self.ratings[item]
            timestamp = self.timestamps[item]
            dayofweek = self.dayofweek[item]

            return {
                "user": torch.tensor(user, dtype=torch.long),
                "movie": torch.tensor(movie, dtype=torch.long),
                "rating": torch.tensor(rating, dtype=torch.float),
                "timestamp": torch.tensor(timestamp, dtype=torch.float),
                "dayofweek": torch.tensor(dayofweek, dtype=torch.long),

            }

    df = pd.concat(all_dataframes, ignore_index=True)

    # Convert Date to Timestamp
    df['Date'] = pd.to_datetime(df['Date'])

    df['Timestamp'] = df['Date'].apply(lambda x: x.timestamp())
    df['DayOfWeek'] = df['Date'].dt.dayofweek



    # Map CustomerID to a new set of IDs starting from 0
    unique_customer_ids = sorted(df.CustomerID.unique())
    customer_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_customer_ids)}
    df["CustomerID"] = df["CustomerID"].map(customer_id_mapping)

    # Adjust MovieIDs to zero-based indexing
    df["MovieID"] = df["MovieID"] - 1

    # Fill missing timestamps if there are any
    df["Timestamp"] = df["Timestamp"].fillna(df["Timestamp"].median())
    scaler = StandardScaler()

    if pretrain:
        df["Timestamp"] = scaler.fit_transform(df["Timestamp"].values.reshape(-1, 1))
        dataset = NetflixDataset(
            users=df.CustomerID.values,
            movies=df.MovieID.values,
            ratings=df.Rating.values,
            timestamps=df.Timestamp.values,
            dayofweek=df.DayOfWeek.values,
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader, df
    else:
        # Split data into train and test sets
        df_train, df_test = model_selection.train_test_split(
            df, test_size=0.2, random_state=42, stratify=df.Rating.values
        )

        # Fit and transform timestamps using StandardScaler

        df_train["Timestamp"] = scaler.fit_transform(df_train["Timestamp"].values.reshape(-1, 1))
        df_test["Timestamp"] = scaler.transform(df_test["Timestamp"].values.reshape(-1, 1))

        # Define Dataset class

            # Create train and test datasets
        train_dataset = NetflixDataset(
                users=df_train.CustomerID.values,
                movies=df_train.MovieID.values,
                ratings=df_train.Rating.values,
                timestamps=df_train.Timestamp.values,
                dayofweek=df_train.DayOfWeek.values,

        )
        test_dataset = NetflixDataset(
                users=df_test.CustomerID.values,
                movies=df_test.MovieID.values,
                ratings=df_test.Rating.values,
                timestamps=df_test.Timestamp.values,
                dayofweek=df_test.DayOfWeek.values,

            )

            # Create DataLoaders with a reduced batch size and appropriate number of workers
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, df