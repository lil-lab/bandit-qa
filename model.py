import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel
import torch


class BertForQuestionAnswering(nn.Module):
    def __init__(self, model_type: str):
        super(BertForQuestionAnswering, self).__init__()
        if 'bert-' in model_type:
            self.bert = BertModel.from_pretrained(model_type)
        else:
            raise ValueError('Model type!')

        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # [N, L, H] => [N, L, 2]

    def forward(self, batch, return_prob=False, **kwargs):
        '''
        each batch is a list of 5 items (training) or 3 items (inference)
            - input_ids: token id of the input sequence
            - attention_mask: mask of the sequence (1 for present, 0 for blank)
            - token_type_ids: indicator of type of sequence.
            -      e.g. in QA, whether it is question or document
            - (training) start_positions: list of start positions of the span
            - (training) end_positions: list of end positions of the span
        '''

        input_ids, attention_masks, token_type_ids = batch[:3]
        # pooler_output, last_hidden_state
        if 'distil' in self.bert.config._name_or_path:
            output = self.bert(
                input_ids=input_ids,
                # NOTE token_types_ids is not an argument for distilbert
                # token_type_ids=token_type_ids,
                attention_mask=attention_masks)
        else:
            output = self.bert(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_masks)
        sequence_output = output.last_hidden_state
        logits = self.qa_outputs(sequence_output)  # (bs, max_input_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_input_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_input_len)

        if len(batch) == 5:
            start_positions, end_positions = batch[3:]
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, None

        elif len(batch) == 3:
            if not return_prob:
                return start_logits, end_logits
            else:
                return torch.softmax(start_logits, dim=-1), torch.softmax(end_logits, dim=-1)

        else:
            raise NotImplementedError()
