import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import dill
from torchvision import transforms
from model import EncoderCNN, DecoderTransformer
from vocab import Vocabulary

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_LEN = 20
MODEL_PATH_ENCODER = "checkpoint/encoder_final.pth"
MODEL_PATH_DECODER = "checkpoint/decoder_final.pth"
TOKENIZER_PATH = "tokenizer.pkl"

# === Load Tokenizer ===
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = dill.load(f)

# === Load Models ===
encoder = EncoderCNN(embed_size=EMBED_SIZE).to(DEVICE)
decoder = DecoderTransformer(
    embed_size=EMBED_SIZE,
    vocab_size=tokenizer.vocab_size,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=MAX_LEN
).to(DEVICE)

encoder.load_state_dict(torch.load(MODEL_PATH_ENCODER, map_location=DEVICE))
decoder.load_state_dict(torch.load(MODEL_PATH_DECODER, map_location=DEVICE))
encoder.eval()
decoder.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Caption Prediction ===
def create_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1).to(DEVICE)

def predict_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = encoder(image_tensor)
        caption_input = torch.tensor([[tokenizer.word2idx["<START>"]]], device=DEVICE)

        for _ in range(MAX_LEN):
            tgt_mask = create_mask(caption_input.size(1))
            outputs = decoder(features, caption_input, tgt_mask)
            next_token_logits = outputs[:, -1, :]
            next_token_logits[0][tokenizer.word2idx["<UNK>"]] = -float('inf')
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            caption_input = torch.cat([caption_input, next_token], dim=1)
            if next_token.item() == tokenizer.word2idx["<END>"]:
                break

        output_ids = caption_input[0].cpu().tolist()
        words = [tokenizer.idx2word[idx] for idx in output_ids
                 if idx not in [tokenizer.word2idx["<START>"], tokenizer.word2idx["<END>"], tokenizer.word2idx["<PAD>"]]]
        return " ".join(words)

# === Animate Caption ===
def animate_caption(text, index=0):
    if index < len(text):
        caption_text.insert(tk.END, text[index])
        caption_text.yview_moveto(1.0)
        root.after(30, animate_caption, text, index + 1)

# === Upload & Process ===
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
    if not file_path:
        return

    caption_text.delete("1.0", tk.END)
    caption_text.insert(tk.END, "🔍 Looking at the floor plan...\n")
    root.update_idletasks()

    # Display image
    img = Image.open(file_path)
    img.thumbnail((360, 360))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Animation
    root.after(1000, lambda: caption_text.insert("end", "🤖 Thinking...\n"))
    root.after(2000, lambda: caption_text.insert("end", "✨ Almost ready...\n"))

    def generate_and_display_caption():
        try:
            caption = predict_caption(file_path)
            caption_text.delete("1.0", tk.END)
            animate_caption(f"📝 {caption}")
        except Exception as e:
            caption_text.delete("1.0", tk.END)
            caption_text.insert("1.0", f"❌ Error: {e}")

    root.after(3000, generate_and_display_caption)

# === Room Legend ===
room_legend = {
    "Balcony": "#90EE90",
    "Bathroom": "#ADD8E6",
    "Living Room": "#FFD700",
    "Kitchen": "#FFA07A",
    "Bedroom": "#FFB6C1",
    "Common Room": "#f5f0b5"  # <-- Added this line (light yellow)
}

# === STYLING ===
BG_COLOR = "#f7f9fc"
TITLE_COLOR = "#3a506b"
ACCENT_COLOR = "#4b7bec"
BUTTON_COLOR = "#3fc1c9"
TEXT_BG = "#ffffff"
TEXT_FG = "#333333"
FONT = ("Helvetica", 12)

# === GUI Setup ===
root = tk.Tk()
root.title("🏠 Floor Plan Captioning AI")
root.geometry("900x600")
root.configure(bg=BG_COLOR)

title_label = tk.Label(root, text="🧠 Floor Plan Captioning AI", font=("Helvetica", 18, "bold"), bg=BG_COLOR, fg=TITLE_COLOR)
title_label.pack(pady=15)

main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(pady=10, fill="both", expand=True)

img_label = tk.Label(main_frame, bg=BG_COLOR)
img_label.pack(side="left", padx=20)

legend_frame = tk.Frame(main_frame, bg=TEXT_BG, relief="groove", borderwidth=2)
legend_frame.pack(side="right", padx=20, fill="y")
legend_title = tk.Label(legend_frame, text="🏷️ Room Legend", font=("Helvetica", 13, "bold"), bg=TEXT_BG, fg=TITLE_COLOR)
legend_title.pack(pady=8)

for room, color in room_legend.items():
    room_label = tk.Label(legend_frame, text=room, font=FONT, bg=color, width=20, relief="ridge")
    room_label.pack(pady=2, padx=10)

upload_btn = tk.Button(root, text="📂 Upload Floor Plan", font=("Helvetica", 13, "bold"), command=upload_image, bg=BUTTON_COLOR, fg="white", activebackground="#2f89fc", relief="raised", padx=15, pady=5)
upload_btn.pack(pady=15)

caption_frame = tk.Frame(root, bg=BG_COLOR)
caption_frame.pack(pady=10, fill="both", expand=True)

caption_scrollbar = tk.Scrollbar(caption_frame, orient="vertical")
caption_scrollbar.pack(side="right", fill="y")

caption_text = tk.Text(caption_frame, font=FONT, bg=TEXT_BG, fg=TEXT_FG, wrap="word", height=6, width=70, yscrollcommand=caption_scrollbar.set, relief="flat", bd=2)
caption_text.pack(side="left", fill="both", expand=True)

caption_scrollbar.config(command=caption_text.yview)

root.mainloop()
