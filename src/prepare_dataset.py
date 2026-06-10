import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

df = pd.read_csv(csv_path)

# column names
image_col = "image"
label_col = "label"
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df[label_col],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df[label_col],
    random_state=42
)

base_output = "dataset"

classes = df[label_col].unique()

for split in ["train", "val", "test"]:
    for c in classes:
        os.makedirs(os.path.join(base_output, split, c), exist_ok=True)


def copy_images(dataframe, split):

    for _, row in dataframe.iterrows():

        img_name = row[image_col] + ".jpg"
        label = row[label_col]

        src = os.path.join(image_dir, img_name)
        dst = os.path.join(base_output, split, label, img_name)

        if os.path.exists(src):
            shutil.copy(src, dst)


copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print("Dataset prepared successfully.")