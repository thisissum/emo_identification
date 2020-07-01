import torch
from torch import nn
from modules.layers import MultiKernelConv1d
from transformers import BertModel

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.emb_layer = nn.Embedding(
            num_embeddings=24000,
            embedding_dim=config.emb_dim,
            padding_idx=0
        )
        self.bert_encoder = BertModel.from_pretrained(config.bert_path)
        self.conv_encoder = MultiKernelConv1d(
            in_channels=config.emb_dim,
            out_channels=config.hidden_dim,
            kernel_size=config.kernel_size,
            dilation=config.dilation
        )

        self.max_pooler = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pooler = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(config.bert_dim*2 + config.hidden_dim*2, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )

    def forward(self, input_ids, mask, seg_ids):
        # BERT part
        hidden, cls = self.bert_encoder(input_ids, attention_mask=mask, token_type_ids=seg_ids.long())
        bert_max_pooler = self.max_pooler(hidden.permute(0,2,1)).squeeze()
        # CNN part
        emb = self.emb_layer(input_ids)
        conv_hidden = self.conv_encoder(emb, mask=mask)
        conv_max_pooler = self.max_pooler(conv_hidden.permute(0,2,1)).squeeze()
        conv_avg_pooler = self.avg_pooler(conv_hidden.permute(0,2,1)).squeeze()
        # Concat all pooler vertor
        output_pooler = torch.cat([bert_max_pooler, cls, conv_max_pooler, conv_avg_pooler], dim=-1)
        score = self.fc(output_pooler)
        return score