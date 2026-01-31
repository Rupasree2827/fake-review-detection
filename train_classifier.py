import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from review_classifier import ReviewClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np

class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.reviews = data['review'].values
        self.labels = data['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.reviews[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }


# ----------------- Load dataset -----------------
df = pd.read_csv("fake_reviews.csv")

# Rename if dataset has custom column names
df.rename(columns={"your_review_column": "review", "your_label_column": "label"}, inplace=True)

train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ReviewDataset(train_df, tokenizer)
val_dataset = ReviewDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ----------------- Model Setup -----------------
model = ReviewClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()  # better than BCELoss for raw logits

# ----------------- Training Loop -----------------
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)  # shape (batch,1)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)  # raw logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

# ----------------- Evaluation -----------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4)

print(f"✅ Validation Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Save model
torch.save(model.state_dict(), 'bert_fake_review.pt')

# Save validation predictions for Streamlit
val_results = pd.DataFrame({
    "review": val_df["review"].values,
    "true_label": np.array(y_true).flatten(),
    "pred_label": np.array(y_pred).flatten()
})
val_results.to_csv("reviews.csv", index=False)
print("📂 Saved model and validation predictions.")
