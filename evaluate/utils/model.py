import torch
import torch.nn as nn

#hf models
from transformers import AutoModel, PreTrainedModel

from transformers.modeling_outputs import BaseModelOutputWithPooling


class GeneralModelForSequenceClassification(PreTrainedModel):
    def __init__(self, config, model_name, weights, pooling="average", train_dataset=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights = weights
        self.pooling = pooling
        self.model_name = model_name
        self.train_dataset = train_dataset

        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
    
        if self.model_name == "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1":
            outputs = self.encoder(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                            )
        elif self.model_name == "jxm/cde-small-v1":
            pooled_output = self.encoder.second_stage_model(input_ids, #second_stage_model
                                attention_mask=attention_mask,
                                #dataset_input_ids = None,
                                #dataset_attention_mask = None,
                                #'dataset_input_ids' and 'dataset_attention_mask
                                dataset_embeddings=generate_dataset_embeddings(self.train_dataset),
                                )

            # print(outputs)
            # exit()
            outputs = BaseModelOutputWithPooling(
                last_hidden_state=None,
                pooler_output=pooled_output
            )
            #torch.Size([64, 768])
        else:
            outputs = self.encoder(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict
                                )
            #torch.Size([77, 64, 1024])
        
        if self.model_name != "jxm/cde-small-v1":
            if self.pooling == "average":
                attention_mask = attention_mask.unsqueeze(-1)
                pooled_output = torch.sum(outputs[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            elif self.pooling == "cls_nopool":
                pooled_output = outputs[0][:, 0, :]
            elif self.pooling == "cls":
                pooled_output = outputs[1] if len(outputs) > 1 else outputs[0][:, 0, :]
            else:
                raise ValueError("Choose args.pooling from ['cls', 'cls_nopool', 'average']")

            pooled_output = self.dropout(pooled_output)
            pooled_output = pooled_output.to(torch.float32)
        # print(pooled_output.shape)
        seq_logits = self.classifier(pooled_output)

        # print(outputs)

        outputs = (seq_logits,) + outputs[2:] 
        if labels is not None:
            seq_loss_fct = nn.CrossEntropyLoss().cuda()
            loss = seq_loss_fct(seq_logits, labels)
            outputs = (loss,) + outputs

        return outputs