import torch
from torch.utils.data import Dataset

# prefix lm https://github.com/bigscience-workshop/Megatron-DeepSpeed/issues/21
#


class LLMDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenizer(self.data[idx])
