import torch
from transformers import BertTokenizer

class Processor(object):
    def __init__(self, name=None):
        self.name = name

    def __call__(self, line):
        pass

class PureTextProcessor(Processor):
    def __init__(self, bert_path, seq_len):
        super(PureTextProcessor, self).__init__(name='pure_text')
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.seq_len = seq_len

    def __call__(self, line):
        sentence = line['text']
        if 'target' in line.index:
            target = torch.tensor(line['target'])
        tk_output = self.tokenizer.encode_plus(
            sentence,
            max_length=self.seq_len,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        ids, mask, seg = tk_output['input_ids'], tk_output['attention_mask'], tk_output['token_type_ids']
        if 'target' in line.index:
            target = torch.tensor(line['target'])
            return ids, mask, seg, target
        return ids, mask, seg

