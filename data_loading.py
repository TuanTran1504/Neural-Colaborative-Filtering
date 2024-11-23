import pandas as pd
import os

train_data_dir="training_data"
data=[]
for filename in os.listdir(train_data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(train_data_dir,filename),"r") as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    movie_id=int(line[:-1])
                else:
                    customer_id, rating, date = line.strip().split(',')
                    data.append([movie_id, int(customer_id), int(rating), date])
df = pd.DataFrame(data, columns=["MovieID", "CustomerID", "Rating", "Date"])

df[0:500000].to_csv("processed_data_test.csv", index=False)

