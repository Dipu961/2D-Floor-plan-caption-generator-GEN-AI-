class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}

        # Add special tokens
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        self.add_word(self.pad_token)
        self.add_word(self.start_token)
        self.add_word(self.end_token)
        self.add_word(self.unk_token)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
                if self.word_freq[word] == self.freq_threshold:
                    self.add_word(word)

    def numericalize(self, text, max_len=20):
        tokens = self.tokenize(text)
        ids = [self.word2idx[self.start_token]] + \
              [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in tokens[:max_len - 2]] + \
              [self.word2idx[self.end_token]]

        if len(ids) < max_len:
            ids += [self.word2idx[self.pad_token]] * (max_len - len(ids))
        return ids

    def tokenize(self, text):
        return text.lower().strip().split()

    def __len__(self):
        return len(self.word2idx)
