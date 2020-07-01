import torch
import pandas as pd
from processor import *

def get_dataloader(config, data):
    if config.processor_name == 'pure_text':
        processor = PureTextProcessor(config.bert_path, config.seq_len)

    if config.file_type == 'csv':
        dataset = CSVDataset(data, processor)

    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    return loader


class CSVDataset(torch.utils.data.Dataset):
    """
    load csv data with pd.DataFrame
    """
    def __init__(self, data, processor):
        super(CSVDataset, self).__init__()
        self.data = data
        self.processor = processor

    def __getitem__(self, index):
        return self.processor(self.data.iloc[index])

    def __len__(self):
        return len(self.data)