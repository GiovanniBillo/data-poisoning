import torch
import os

directory = "embeddings"
filename = "train_samples.pth"

filepath = os.path.join(directory, filename)

data = torch.load(filepath, map_location="cpu")  # optional map_location
data_imgs = data['images']
data_labels = data['labels']
print(type(data_imgs), data_imgs.shape)
print(type(data_labels), data_labels.shape)

filename_2 = "valid_samples.pth"

filepath_2 = os.path.join(directory, filename_2)

v_data = torch.load(filepath_2, map_location="cpu")  # optional map_location
v_data_imgs = v_data['images']
v_data_labels = v_data['labels']
print(type(v_data_imgs), v_data_imgs.shape)

print("onto checking embeddings shape...")
embedding1 =  "embeddings_CLEAN_epoch20.pth"
embedding2 =  "embeddings_CLEAN_epoch40.pth"

emb_filepath1= os.path.join(directory, embedding1)
emb_filepath2= os.path.join(directory, embedding2)

emb_data1 = torch.load(emb_filepath1, map_location="cpu")  # optional map_location
emb_data2 = torch.load(emb_filepath2, map_location="cpu")  # optional map_location

# emb_data_imgs1 = emb_data1[0]['fc_out']
all_emb_data_imgs1 = [entry['fc_out'] for entry in emb_data1]
print("for embeddings at epoch 20")
print(type(all_emb_data_imgs1), len(all_emb_data_imgs1), all_emb_data_imgs1[0].shape)

print("for embeddings at epoch 40")
# emb_data_imgs2 = emb_data2[0]['fc_out']
all_emb_data_imgs2 = [entry['fc_out'] for entry in emb_data2]
print(type(all_emb_data_imgs2), len(all_emb_data_imgs2), all_emb_data_imgs2[0].shape)
