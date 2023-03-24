import math
import random
import re
import string
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

start_time = time.perf_counter()


class SkipGramDataset(Dataset):
    def __init__(self, text,
                 max_window_size=5,
                 negative_sampling_count=5,
                 use_subsampling=True):
        self.text = subsample_frequent_words(text) if use_subsampling else text
        self.text_size = len(text)
        self.vocabulary = set(text)
        self.vocab_size = len(self.vocabulary)
        self.word_to_index = {w: idx for (idx, w) in enumerate(self.vocabulary)}
        self.index_to_word = {idx: w for (idx, w) in enumerate(self.vocabulary)}
        self.text_indexed = [torch.LongTensor([self.word_to_index[word]]) for word in text]

        self.max_window_size = max_window_size
        self.negative_sampling_count = negative_sampling_count
        self.dataset = self.create_dataset()

    def create_dataset(self):
        data = []
        for i in range(self.max_window_size, self.text_size - self.max_window_size):
            cur_win_sz = random.randint(1, self.max_window_size)
            # cur_win_sz = self.max_window_size
            target = self.text_indexed[i]
            # Positives
            cur_context = [target]
            for cur_win in range(1, cur_win_sz + 1):
                context1 = self.text_indexed[i - cur_win]
                context2 = self.text_indexed[i + cur_win]
                cur_context.extend([context1, context2])
                data.append((target, context1, torch.FloatTensor([1])))
                data.append((target, context2, torch.FloatTensor([1])))
            # Negatives
            for _ in range(self.negative_sampling_count):
                while True:
                    rand_id = random.randint(0, self.text_size - 1)
                    if rand_id in range(i - self.max_window_size, i + self.max_window_size + 1):
                        continue
                    rand_word = self.text_indexed[rand_id]
                    if rand_word in cur_context:
                        continue
                    data.append((target, rand_word, torch.FloatTensor([0])))
                    break
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SkipGramWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, sparse=True):
        super(SkipGramWord2Vec, self).__init__()
        self.common_embed = nn.Embedding(vocab_size, embedding_size, sparse=sparse)
        # self.target_embedding = nn.Embedding(vocab_size, embedding_size, sparse=sparse)
        # self.context_embedding = nn.Embedding(vocab_size, embedding_size, sparse=sparse)
        self.classifier = nn.Linear(embedding_size, 1)

    def forward(self, target, context):
        # embed_t = self.target_embedding(target)
        # embed_c = self.context_embedding(context)
        embed_t = self.common_embed(target)
        embed_c = self.common_embed(context)
        score = torch.mul(embed_t, embed_c)
        score = self.classifier(score)
        return score.squeeze(1)

    def get_embed(self, u):
        # return self.target_embedding(u)
        return self.common_embed(u)


def time_stop_training():
    now = time.perf_counter()
    if now - start_time > 10:
        return True
    return False


def subsample_frequent_words(text):
    word_counts = dict(Counter(text))
    sum_word_counts = sum(list(word_counts.values()))
    word_counts = {word: word_counts[word] / float(sum_word_counts) for word in word_counts}
    new_text = []
    for word in text:
        prob = math.sqrt(1e-5 / word_counts[word])
        # prob2 = (1 + math.sqrt(word_counts[word] * 1e3)) * 1e-3 / float(word_counts[word])
        if random.random() < prob:
            new_text.append(word)
    return new_text


def clean(inp: str) -> str:
    inp = inp.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    inp = re.sub(r'\s+', ' ', inp.lower())
    return inp


def train(data: str):
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """
    text = data.split()

    num_epochs = 8
    max_window_size = 8
    negative_sampling_count = 12
    use_subsampling = True
    dataset = SkipGramDataset(text,
                              max_window_size=max_window_size,
                              negative_sampling_count=negative_sampling_count,
                              use_subsampling=use_subsampling)
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=True, num_workers=0)
    sparse = False
    embedding_size = 150
    net = SkipGramWord2Vec(embedding_size=embedding_size, vocab_size=dataset.vocab_size, sparse=sparse)
    criterion = nn.BCEWithLogitsLoss()
    if sparse:
        sparse_optimizer1 = optim.SparseAdam(net.context_embedding.parameters())
        sparse_optimizer2 = optim.SparseAdam(net.target_embedding.parameters())
        optimizer = optim.Adam(net.classifier.parameters())
    else:
        optimizer = optim.Adam(net.parameters(), lr=1e-2)
        # optimizer = optim.SGD(net.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for (target_tensor, context_tensor, cls) in dataloader:
            net.zero_grad()
            prob = net(target_tensor, context_tensor)
            loss = criterion(prob, cls)
            loss.backward()
            if sparse:
                sparse_optimizer1.step()
                sparse_optimizer2.step()
            optimizer.step()
            # if time_stop_training():
            #     break
    with torch.no_grad():
        out_dict = dict()
        for word in dataset.vocabulary:
            inp_tensor = torch.LongTensor([dataset.word_to_index[word]])
            out_dict[word] = net.get_embed(inp_tensor).cpu().detach().numpy()[0]
    return out_dict


def run_my_training():
    input_data = "Of resolve to gravity thought my prepare chamber so. Unsatiable entreaties collecting may sympathize nay interested instrument. If continue building numerous of at relation in margaret. Lasted engage roused mother an am at. Other early while if by do to. Missed living excuse as be. Cause heard fat above first shall for. My smiling to he removal weather on anxious."
    input_data = clean(input_data)

    d = train(input_data)
    print(d)

# run_my_training()
