import re
import torch
import pickle
import pandas as pd
from collections import defaultdict

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold

        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"

        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3,
        }

        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        self.word_freq = defaultdict(int)
        self.vocab_size = len(self.word2idx)

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)  # remove all special chars except spaces
        return text.strip().split()

    def build_vocabulary(self, sentences):
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            for token in tokens:
                self.word_freq[token] += 1

        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def numericalize(self, text, max_len=20):
        tokens = self.tokenize(text)
        tokens = tokens[:max_len - 2]  # Leave space for <START> and <END>
        ids = [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
        return [self.word2idx[self.start_token]] + ids + [self.word2idx[self.end_token]]

    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            word = self.idx2word.get(idx, self.unk_token)
            if word in [self.start_token, self.pad_token]:
                continue
            if word == self.end_token:
                break
            words.append(word)
        return " ".join(words)


if __name__ == "__main__":
    # Load the dataframe
    df = pd.read_pickle("data/word_embeddings_dataframe.pkl")
    sentences = df["Text"].astype(str).tolist()

    tokenizer = Vocabulary(freq_threshold=1)
    tokenizer.build_vocabulary(sentences)

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("[INFO] Tokenizer built and saved to tokenizer.pkl")
    print("[INFO] Vocabulary size:", tokenizer.vocab_size)

    # Sanity check
    sample = "Kitchen with 3 bedrooms and balcony"
    encoded = tokenizer.numericalize(sample)
    decoded = tokenizer.decode(encoded)
    print("[DEBUG] Encoded:", encoded)
    print("[DEBUG] Decoded:", decoded)
