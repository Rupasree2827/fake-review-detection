import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class ReviewClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        config = BertConfig.from_pretrained("bert-base-uncased")

        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        print("✅ Model max_position_embeddings:", self.bert.config.max_position_embeddings)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()  # ✅ Added sigmoid layer

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # ✅ Convert logits → probability
        prob = self.sigmoid(logits)
        return prob  # Always returns probability now


def load_model():
    import torch
    from transformers import BertTokenizer
    from review_classifier import ReviewClassifier

    device = torch.device("cpu")  # since you're using +cpu build

    # Initialize empty model
    model = ReviewClassifier().to_empty(device=device)

    # Load weights directly to device
    model.load_state_dict(torch.load("bert_fake_review.pt", map_location=torch.device('cpu')))
    model.to(device)

    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, device
