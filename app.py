import streamlit as st
import torch
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from review_classifier import load_model
from logger import log_review, get_banned_users
from email_utils import send_fake_review_alert


# ✅ Load model
@st.cache_resource
def get_model():
    return load_model()

model, tokenizer, device = get_model()

# ✅ Load validation dataset predictions (saved from training)
@st.cache_data
def load_validation_metrics():
    try:
        df = pd.read_csv("reviews.csv")  # created during training
        labels = df["true_label"].tolist()
        preds = df["pred_label"].tolist()

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        return acc, prec, rec, f1, pd.DataFrame(report).transpose()
    except Exception:
        return None, None, None, None, None


# ----------------- Streamlit Tabs -----------------
st.title("🕵️ Fake Review Detector")

tab1, tab2 = st.tabs(["🔍 Detect Review", "📊 Model Performance"])

# ----------------- Detect Review -----------------
with tab1:
    review = st.text_area("Enter your review:")

    banned_users = get_banned_users()
    user_id = "guest"

    if user_id in banned_users:
        st.error("❌ You are banned from submitting reviews.")
    else:
        if st.button("Detect"):
            if not review.strip():
                st.error("⚠️ Please enter a valid review (not empty).")
            else:
                tokens = tokenizer(
                    review,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)

                input_ids = torch.clamp(input_ids, max=tokenizer.vocab_size - 1)

                with torch.no_grad():
                    token_type_ids = torch.zeros_like(input_ids)
                    logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                    prob = torch.sigmoid(logits).item()
                    label = "🛑 Fake Review" if prob > 0.5 else "✅ Genuine Review"

                    st.markdown(f"**Prediction:** {label} (probability: {prob:.2f})")

                    log_review(user_id=user_id, review_text=review, score=prob, label=label)

                    if prob > 0.5:
                        send_fake_review_alert(
                            review_text=review,
                            score=prob,
                            user_id=user_id,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )

# ----------------- Model Performance -----------------
with tab2:
    st.subheader("📊 Model Evaluation Results (Validation Set)")
    acc, prec, rec, f1, report_df = load_validation_metrics()

    if acc is None:
        st.error("⚠️ Validation metrics not found. Please run training first.")
    else:
        st.metric("✅ Accuracy", f"{acc:.2f}")
        st.metric("📏 Precision", f"{prec:.2f}")
        st.metric("📢 Recall", f"{rec:.2f}")
        st.metric("⭐ F1 Score", f"{f1:.2f}")

        st.dataframe(report_df)
