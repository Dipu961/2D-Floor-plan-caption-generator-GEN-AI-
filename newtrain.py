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

# =================== Configuration ===================
IMAGE_DIR = 'data/human_annotated_images'
DATAFRAME_PATH = 'data/word_embeddings_dataframe.pkl'
TOKENIZER_PATH = 'tokenizer.pkl'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
EMBED_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_LEN = 20

# =================== Load Tokenizer ===================
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("[INFO] Tokenizer loaded. Vocabulary size:", tokenizer.vocab_size)

# =================== Load DataFrame ===================
df = pd.read_pickle(DATAFRAME_PATH)
print("[INFO] Loaded dataframe:", df.shape)

# =================== Dataset and DataLoader ===================
dataset = FloorPlanDataset(df, IMAGE_DIR, tokenizer, max_len=MAX_LEN, use_bert=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# =================== Model Initialization ===================
encoder = EncoderCNN(embed_size=EMBED_SIZE).to(DEVICE)
decoder = DecoderTransformer(embed_size=EMBED_SIZE, vocab_size=tokenizer.vocab_size,
                             num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<PAD>"])
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

# =================== Create Checkpoint Directory ===================
os.makedirs("checkpoint", exist_ok=True)

print("[INFO] Starting training...")

# =================== Masking Function ===================
def create_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

# =================== Training Loop ===================
for epoch in range(EPOCHS):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    total_loss = 0

    for i, (images, captions) in loop:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        # Prepare inputs and targets
        tgt_inputs = captions[:, :-1]
        tgt_outputs = captions[:, 1:]
        tgt_mask = create_mask(tgt_inputs.size(1)).to(DEVICE)

        # Forward pass
        features = encoder(images)
        outputs = decoder(features, tgt_inputs, tgt_mask)

        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_outputs.reshape(-1))
        total_loss += loss.item()

        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    # Save model per epoch
    encoder_checkpoint_path = f"checkpoint/encoder_epoch{epoch+1}.pth"
    decoder_checkpoint_path = f"checkpoint/decoder_epoch{epoch+1}.pth"
    torch.save(encoder.state_dict(), encoder_checkpoint_path)
    torch.save(decoder.state_dict(), decoder_checkpoint_path)
    print(f"[INFO] Checkpoint saved: {encoder_checkpoint_path}, {decoder_checkpoint_path}")
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# =================== Save Final Models ===================
torch.save(encoder.state_dict(), "checkpoint/encoder_final.pth")
torch.save(decoder.state_dict(), "checkpoint/decoder_final.pth")
print("[INFO] Training completed. Final models saved.")