# TODO List

1. Tokenizezi iar, dar trb modificat label-ul: PF_* -> PF; PJ_* -> PJ; STAT_* -> STAT
2. Antrenezi modelul pentru cele 4 clase (PF, PJ, STAT, Locatie)
3. Modific arhitectura, si adaugi un cap sa desparti mai departe clasele. *
* aici trebuie testat, ca e cam aiurea din 4 neuroni sa transformi in 10.
https://huggingface.co/docs/transformers/training
4. Antrenezi iar



from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification('bert-base-uncased', num_labels=2)
# Start your own training
or if you want to write your own as requested with a custom classifier head

import torch.nn as nn
from transformers import AutoModel
class PosModel(nn.Module):
    def __init__(self):
        super(PosModel, self).__init__()
        
        self.base_model = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 2) # output features from bert is 768 and 2 is ur number of labels
        
    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        # You write you new head here
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)
        
        return outputs

model = PosModel()
model.to('cuda')