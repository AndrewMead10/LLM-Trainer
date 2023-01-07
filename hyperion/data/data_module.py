import pytorch_lightning as pl
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import LLMDataset
from tokenizers import Tokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        if cfg.dataset == 'wikitext':
            self.dataset = load_dataset('wikitext')

        self.tokenizer = Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_dataset = LLMDataset(self.dataset['train'], self.tokenizer)

        self.val_dataset = LLMDataset(
            self.dataset['validation'], self.tokenizer)

        self.test_dataset = LLMDataset(self.dataset['test'], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, shuffle=True, collate_fn=pad_sequence)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, shuffle=False, collate_fn=pad_sequence)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, shuffle=False, collate_fn=pad_sequence)
