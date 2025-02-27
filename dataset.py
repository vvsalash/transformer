import torch
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


def dataset_iterator(path: str):
    with open(path) as texts:
        for text in texts:
            yield text.split()


def build_vocab(path: str, min_freq: int):
    return build_vocab_from_iterator(
        dataset_iterator(path),
        specials=['<pad>', '<unk>', '<bos>', '<eos>'],
        min_freq=min_freq
    )


def build_dataloaders(path: str, vocab_en, vocab_de, batch_size: int = 256):
    train_en_tokens, val_en_tokens, train_de_tokens, val_de_tokens = [], [], [], []
    for text in dataset_iterator(f'{path}/data/train.de-en.en'):
        tokens = [2] + [vocab_en[word] if word in vocab_en else vocab_en['<unk>'] for word in text] + [3]
        train_en_tokens += [tokens]

    for text in dataset_iterator(f'{path}/data/train.de-en.de'):
        tokens = [2] + [vocab_de[word] if word in vocab_de else vocab_de['<unk>'] for word in text] + [3]
        train_de_tokens += [tokens]

    for text in dataset_iterator(f'{path}/data/val.de-en.en'):
        tokens = [2] + [vocab_en[word] if word in vocab_en else vocab_en['<unk>'] for word in text] + [3]
        val_en_tokens += [tokens]

    for text in dataset_iterator(f'{path}/data/val.de-en.de'):
        tokens = [2] + [vocab_de[word] if word in vocab_de else vocab_de['<unk>'] for word in text] + [3]
        val_de_tokens += [tokens]

    max_length = 64

    tokenized_en_train = torch.full((len(train_en_tokens), max_length),
                                    vocab_en['<pad>'],
                                    dtype=torch.int32)
    for i, tokens in enumerate(train_en_tokens):
        length = min(len(tokens), max_length)
        tokenized_en_train[i, :length] = torch.tensor(tokens[:length])

    tokenized_de_train = torch.full((len(train_de_tokens), max_length),
                                    vocab_de['<pad>'],
                                    dtype=torch.int32)
    for i, tokens in enumerate(train_de_tokens):
        length = min(len(tokens), max_length)
        tokenized_de_train[i, :length] = torch.tensor(tokens[:length])

    tokenized_en_val = torch.full((len(val_en_tokens), max_length),
                                  vocab_en['<pad>'],
                                  dtype=torch.int32)
    for i, tokens in enumerate(val_en_tokens):
        length = min(len(tokens), max_length)
        tokenized_en_val[i, :length] = torch.tensor(tokens[:length])

    tokenized_de_val = torch.full((len(val_de_tokens), max_length),
                                  vocab_de['<pad>'],
                                  dtype=torch.int32)
    for i, tokens in enumerate(val_de_tokens):
        length = min(len(tokens), max_length)
        tokenized_de_val[i, :length] = torch.tensor(tokens[:length])

    train_dataset = TensorDataset(tokenized_de_train, tokenized_en_train)
    val_dataset = TensorDataset(tokenized_de_val, tokenized_en_val)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader


class Batch:
    def __init__(self, source_data, target_data=None, padding_token: int = 0) -> None:
        self.source_data = source_data
        self.source_mask = (source_data != padding_token).unsqueeze(-2)

        if target_data is not None:
            self.input_target_data = target_data[:, :-1]
            self.expected_target_data = target_data[:, 1:]
            self.target_mask = self.create_target_mask(self.input_target_data, padding_token)
            self.valid_tokens_count = (self.expected_target_data != padding_token).data.sum()

    def create_target_mask(self, target_data, padding_token: int):
        padding_mask = (target_data != padding_token).unsqueeze(-2)
        future_mask = Variable(
            self.create_future_mask(target_data.size(-1)).type_as(padding_mask.data)
        )
        return padding_mask & future_mask

    @staticmethod
    def create_future_mask(sequence_length: int):
        mask_shape = (1, sequence_length, sequence_length)
        future_mask = np.triu(np.ones(mask_shape), k=1).astype('uint8')
        return torch.from_numpy(future_mask) == 0
