import os
import pandas as pd

base_path = "/Users/sahajdang/Documents/dataset"

image_folder = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_labels.csv")

df = pd.read_csv(csv_path)

# Remove extension from image files
available_images = [f.replace(".jpg", "") for f in os.listdir(image_folder)]

df_filtered = df[df["image"].isin(available_images)]

print("Original CSV size:", len(df))
print("Filtered CSV size:", len(df_filtered))

df_filtered.to_csv(os.path.join(base_path, "classification_filtered.csv"), index=False)