import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, min_freq=2, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_built = False

    def normalize(self, text):
        # Remove extra spaces and normalize punctuation
        return re.sub(r'\s+', ' ', text.strip())

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        most_common = [w for w, c in counter.items() if c >= self.min_freq]
        most_common = most_common[:self.max_vocab_size - len(self.word2idx)]
        for w in most_common:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        self.vocab_built = True

    def tokenize(self, text):
        # Simple whitespace tokenizer
        return self.normalize(text).split(' ')

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]

    def decode(self, ids):
        return ' '.join([self.idx2word.get(i, '<unk>') for i in ids])

    def vocab_size(self):
        return len(self.word2idx)
