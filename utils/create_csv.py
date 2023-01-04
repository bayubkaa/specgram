import pandas as pd
import os
import torch

device = ("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.DataFrame(columns=["img_name","label"])

#dir = "./char_data"
dir = "./data_testing"
dir_list = os.listdir(dir)
dir_list.sort()
print(dir_list)

loc = 0

for i, class_folder in enumerate(dir_list):
    new_class_folder = os.listdir(os.path.join(dir, class_folder))
    for j, file_img in enumerate(new_class_folder):
        print(class_folder, file_img)
        train_df.loc[loc] = ['.'+os.path.join(dir, class_folder, file_img)] + [i]
        loc+=1
train_df.to_csv (r'data_annotations_testing.csv', index = False, header=True)
