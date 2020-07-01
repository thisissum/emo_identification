import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import argparse
from models import BertCNN
from sklearn.model_selection import KFold
from utils.loaders import get_dataloader
from torch_train.trainer import Trainer, seed_everything
from torch_train.metrics import F1Metric
from torch_train.callbacks import EarlyStoping


def args_parser():
    """Set all config needed
    """
    parser = argparse.ArgumentParser()
    # set hyperparameters
    parser.add_argument("--bert_path", default='./embeddings', type=str)
    parser.add_argument("--seq_len", default=140, type=int)
    parser.add_argument("--train_path", default='./data/train.csv', type=str)
    parser.add_argument("--test_path", default='./data/test.csv', type=str)
    parser.add_argument("--processor_name", default='pure_text', type=str)
    parser.add_argument("--file_type", default='csv', type=str)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--lr_schedule_patience", default=2, type=int)
    parser.add_argument("--early_stop_patience", default=3, type=int)
    parser.add_argument("--checkpoint_path", default='./checkpoint/last_best.pt', type=int)
    parser.add_argument("--num_epoch", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--lr_discount", default=0.5, type=float)
    parser.add_argument("--emb_dim", default=256, type=int)
    parser.add_argument("--kernel_size", default=[2,3,4,5], type=int)
    parser.add_argument("--hiddem_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=3, type=int)
    parser.add_argument("--kfold", default=5, type=int)

    config = parser.parse_args()
    return config


class MyTrainer(Trainer):
    def forward_batch(self, data):
        ids, mask, seg, y_true = data
        y_pred = self.model(ids, mask, seg)
        return y_pred, y_true

    def predict_batch(self, data):
        ids, mask, seg = data
        y_pred = self.model(ids, mask, seg)
        y_out = y_pred.argmax(dim=-1)
        return y_out


def create_trainer(config):
    criterion = nn.CrossEntropyLoss()
    model = BertCNN(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    metric = F1Metric(average='macro')
    callbacks = [ReduceLROnPlateau(optimizer, mode='max', factor=config.lr_discount, patience=config.lr_schedule_patience),
                 EarlyStoping(model, mode='max', patience=config.early_stop_patience, path=config.checkpoint_path)]
    trainer = MyTrainer(device=config.device)
    trainer.build(model, optimizer, criterion, callbacks, metric)
    return trainer


def main(config):
    data = pd.read_csv(config.train_path)
    test_data = pd.read_csv(config.test_path)
    test_loader = get_dataloader(config, test_data)
    kfold = KFold(n_splits=config.kfold)
    outputs = []
    for i, (train_idx, val_idx) in enumerate(kfold.split(data)):
        train_loader = get_dataloader(config, data.iloc[train_idx])
        val_loader = get_dataloader(config, data.iloc[val_idx])
        trainer = create_trainer(config)
        trainer.fit(train_loader, num_epoch=config.num_epoch, validate_loader=val_loader)
        y_pred = trainer.predict(test_loader=test_loader)
        output = y_pred.cpu().tolist()
        outputs.append(output)
    pd.DataFrame(outputs).to_csv('predictions.csv')


if __name__ == '__main__':
    config = args_parser()
    main(config)


