from pytorch_lightning import Trainer
from models.pl_module import PLModule
from models.llms.palm import Transformer
from data.data_module import DataModule
from pytorch_lightning.loggers import WandbLogger


class Config:
    dataset = 'wikitext'
    batch_size = 64
    num_workers = 2
    d_model = 768
    dropout = 0.1
    mlp_ratio = 4
    n_heads = 12
    casual = True
    bias = False
    vocab_size = 50257
    max_len = 2048
    n_layers = 12
    pin_memory = True
    lr = 5e-4


def main():
    cfg = Config()
    model = PLModule(cfg, Transformer(cfg))
    data = DataModule(cfg)
    logger = WandbLogger(project='hyperion', name='init_test')
    trainer = Trainer(gpus=1, max_epochs=1, logger=logger)
    trainer.fit(model, data)
