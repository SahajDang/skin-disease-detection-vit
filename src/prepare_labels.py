import pandas as pd
import os

base_path = "/Users/sahajdang/Documents/dataset"
csv_path = os.path.join(base_path, "classification_filtered.csv")

df = pd.read_csv(csv_path)

# Get class columns
class_columns = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Convert one-hot → label index
df["label"] = df[class_columns].idxmax(axis=1)

# Keep only image + label
df_final = df[["image", "label"]]

df_final.to_csv(os.path.join(base_path, "classification_ready.csv"), index=False)

print("New CSV created with labels.")
print(df_final.head())