import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ResNet normalization
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row["Image_Path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Tokenize and pad caption
        caption = row["Text"]
        caption_ids = self.tokenizer.numericalize(caption, max_len=self.max_len)

        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        if self.use_bert:
            # BERT embeddings are precomputed and stored in the dataframe
            bert_embedding = torch.tensor(row["BERT_Embeddings"], dtype=torch.float)  # shape: (max_len, 768)
            return image, caption_tensor, bert_embedding
        else:
            return image, caption_tensor
