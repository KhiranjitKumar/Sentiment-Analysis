import torch
import torch.nn as nn
from transformers import BertModel

class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim=256, n_layers=1, dropout=0.3):
        super(BERTBiLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=n_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)  

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state 
        
        lstm_out, (hidden, _) = self.lstm(last_hidden_state)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output
