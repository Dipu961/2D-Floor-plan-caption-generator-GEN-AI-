import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
from model import EncoderCNN, DecoderTransformer
from vocab import Vocabulary

# NEW: Make sure image window displays properly
import matplotlib
matplotlib.use('TkAgg')  # <- 'TkAgg' or 'Qt5Agg' or 'MacOSX' depending on OS
import matplotlib.pyplot as plt

# ==== CONFIG ====
IMAGE_PATH = "data/human_annotated_images/20.png"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "checkpoint/encoder_final.pth"
DECODER_PATH = "checkpoint/decoder_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_LEN = 20

# ==== LOAD TOKENIZER ====
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("[INFO] Tokenizer loaded. Vocabulary size:", tokenizer.vocab_size)

# ==== LOAD MODELS ====
encoder = EncoderCNN(embed_size=EMBED_SIZE).to(DEVICE)
decoder = DecoderTransformer(
    embed_size=EMBED_SIZE,
    vocab_size=tokenizer.vocab_size,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=MAX_LEN
).to(DEVICE)

encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
encoder.eval()
decoder.eval()
print("[INFO] Models loaded and ready.")

# ==== IMAGE LOADER ====
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return image_tensor.to(DEVICE), image

# ==== SUBSEQUENT MASK CREATOR ====
def create_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1).to(DEVICE)

# ==== PREDICT CAPTION ====
def predict_caption(image_tensor):
    with torch.no_grad():
        features = encoder(image_tensor)
        caption_input = torch.tensor([[tokenizer.word2idx["<START>"]]], device=DEVICE)

        for _ in range(MAX_LEN):
            tgt_mask = create_mask(caption_input.size(1))
            outputs = decoder(features, caption_input, tgt_mask)
            next_token_logits = outputs[:, -1, :]

            # Avoid predicting <UNK>
            next_token_logits[0][tokenizer.word2idx["<UNK>"]] = -float('inf')

            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            caption_input = torch.cat([caption_input, next_token], dim=1)

            if next_token.item() == tokenizer.word2idx["<END>"]:
                break

        output_ids = caption_input[0].cpu().tolist()
        words = [tokenizer.idx2word[idx] for idx in output_ids
                 if idx not in [tokenizer.word2idx["<START>"], tokenizer.word2idx["<END>"], tokenizer.word2idx["<PAD>"]]]
        return " ".join(words)

# ==== DISPLAY FUNCTION ====
def show_image_with_caption(image, caption):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"📝 {caption}", fontsize=12, wrap=True)
    plt.tight_layout()
    plt.show()

# ==== MAIN ====
if __name__ == "__main__":
    image_tensor, image_raw = load_image(IMAGE_PATH)
    caption = predict_caption(image_tensor)
    print("\n📝 Predicted Caption:", caption)
    show_image_with_caption(image_raw, caption)
