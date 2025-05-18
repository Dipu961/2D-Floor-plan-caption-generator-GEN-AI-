# train.py
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from utils import FloorPlanDataset, collate_fn
from model import EncoderCNN, DecoderTransformer
from vocab import Vocabulary

# Config
IMAGE_DIR = 'data/human_annotated_images'
DATAFRAME_PATH = 'data/word_embeddings_dataframe.pkl'
TOKENIZER_PATH = 'tokenizer.pkl'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
EMBED_SIZE = 512
HIDDEN_SIZE = 512
MAX_LEN = 20

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("[INFO] Tokenizer loaded. Vocabulary size:", tokenizer.vocab_size)

# Load dataframe
df = pd.read_pickle(DATAFRAME_PATH)
print("[INFO] Loaded dataframe:", df.shape)

# Dataset & DataLoader
dataset = FloorPlanDataset(df, IMAGE_DIR, tokenizer, max_len=MAX_LEN, use_bert=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Model setup
encoder = EncoderCNN(embed_size=EMBED_SIZE).to(DEVICE)
decoder = DecoderTransformer(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=tokenizer.vocab_size).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<PAD>"])
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

print("[INFO] Starting training...")

# Training loop
for epoch in range(EPOCHS):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    total_loss = 0

    for i, (images, captions, bert_embeddings) in loop:
        images, captions, bert_embeddings = images.to(DEVICE), captions.to(DEVICE), bert_embeddings.to(DEVICE)
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])
        targets = captions[:, 1:]

        loss = criterion(outputs.view(-1, outputs.size(2)), targets.contiguous().view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

print("[INFO] Training completed.")
