import torch
import torch.nn as nn
import torchvision.models as models
import math


# ================= Encoder ================= #
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # remove last FC layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features  # (batch_size, embed_size)


# ================= Positional Encoding ================= #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ================= Decoder Transformer ================= #
class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, max_len=20):
        super(DecoderTransformer, self).__init__()
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = self._generate_positional_encoding(max_len, embed_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def _generate_positional_encoding(self, max_len, embed_size):
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # shape: (1, max_len, embed_size)

    def forward(self, memory, tgt, tgt_mask):
        batch_size, tgt_len = tgt.size()
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)
        pe = self.positional_encoding[:, :tgt_len, :].to(tgt.device)
        tgt_emb = tgt_emb + pe

        tgt_emb = tgt_emb.permute(1, 0, 2)  # (tgt_len, batch, embed_size)
        memory = memory.unsqueeze(0)        # (1, batch, embed_size)

        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out = out.permute(1, 0, 2)  # (batch, tgt_len, embed_size)
        out = self.fc_out(out)
        return out
    def generate(self, memory, tokenizer, max_len=20, device="cpu"):
        generated = [tokenizer.word2idx["<START>"]]
        memory = memory.unsqueeze(0)  # (1, batch, embed_size)

        for _ in range(max_len):
            tgt = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
            tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_emb = tgt_emb.permute(1, 0, 2)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)

            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            out = out.permute(1, 0, 2)
            out = self.fc_out(out[:, -1, :])  # (batch, vocab_size)
            _, next_token = out.max(1)
            next_token_id = next_token.item()
            generated.append(next_token_id)

            if next_token_id == tokenizer.word2idx["<END>"]:
                break

        return torch.tensor(generated[1:])  # exclude <START>
