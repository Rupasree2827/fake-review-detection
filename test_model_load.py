import torch
from review_classifier import ReviewClassifier

device = torch.device("cpu")

model = ReviewClassifier()
state_dict = torch.load("bert_fake_review.pt", map_location=torch.device('cpu'))

print("✅ Model weights loaded. Keys:")
print(state_dict.keys())
