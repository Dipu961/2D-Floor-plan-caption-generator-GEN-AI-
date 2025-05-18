import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FloorPlanDataset(Dataset):
    def __init__(self, dataframe, image_dir, tokenizer, max_len=20, use_bert=True):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_bert = use_bert

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Pre-trained ResNet normalization
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Use correct image path (already full path in DataFrame)
        image_path = os.path.normpath(row["Image_Path"])  # Fix: avoid double prefix
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Tokenize caption
        caption = row["Text"]
        caption_ids = self.tokenizer.numericalize(caption, max_len=self.max_len)
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        if self.use_bert:
            bert_embedding = torch.tensor(row["BERT_Embeddings"], dtype=torch.float)  # shape: (max_len, 768)
            return image, caption_tensor, bert_embedding
        else:
            return image, caption_tensor


# Custom collate function for DataLoader
def collate_fn(batch):
    if len(batch[0]) == 3:
        images, captions, bert_embeddings = zip(*batch)
        images = torch.stack(images)
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
        bert_embeddings = torch.stack(bert_embeddings)
        return images, captions, bert_embeddings
    else:
        images, captions = zip(*batch)
        images = torch.stack(images)
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
        return images, captions
from PIL import Image
from torchvision import transforms
import torch

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

