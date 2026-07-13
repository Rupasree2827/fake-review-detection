# AI-Based Fake Review Detection System

An AI-powered web application that detects whether an online product review is genuine or fake using Natural Language Processing (NLP) and the BERT transformer model. The application provides real-time predictions through an interactive Streamlit interface and helps improve trust in online review platforms.


## Features

- Detects fake and genuine online reviews
- Uses BERT for text classification
- Real-time review prediction
- User-friendly Streamlit interface
- Automated email alert support
- NLP-based text preprocessing
- Performance evaluation using standard ML metrics


## Tech Stack

Frontend
- Streamlit

Backend
- Python

Machine Learning
- BERT
- Transformers
- Scikit-learn
- Pandas
- NumPy

Libraries
- Streamlit
- Torch
- Transformers
- Scikit-learn
- Pandas
- NumPy


## Project Architecture

User Review
      │
      ▼
Text Preprocessing
      │
      ▼
BERT Model
      │
      ▼
Prediction
      │
      ▼
Streamlit Web Application
      │
      ▼
Result Display


## Project Structure

fake-review-detection/
│
├── dataset/
├── model/
├── notebooks/
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── screenshots/


## Installation

Clone the repository

```bash
git clone https://github.com/Rupasree2827/fake-review-detection.git
```

Move into the project

```bash
cd fake-review-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## How It Works

1. User enters a product review.
2. Text preprocessing is performed.
3. Review is passed to the trained BERT model.
4. Model predicts whether the review is Fake or Genuine.
5. Result is displayed instantly through the Streamlit interface.

---

## Machine Learning Workflow

- Data Collection
- Data Cleaning
- Text Preprocessing
- Tokenization
- BERT Model Training
- Model Evaluation
- Streamlit Deployment

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

## Future Enhancements

- Multi-language review detection
- Sentiment Analysis integration
- Review authenticity score
- Cloud deployment
- API integration
- Dashboard for analytics

---

## Screenshots

### Genuine review Output 

<img width="511" height="318" alt="image" src="https://github.com/user-attachments/assets/ea1e2a8a-b8ce-4917-bc92-06d216f45c2c" />


---

### Fake review Output 

<img width="512" height="267" alt="image" src="https://github.com/user-attachments/assets/ba44a09a-86d8-4181-9b89-a78d6c17fb4e" />


---

### Sample mail generated 

<img width="323" height="343" alt="image" src="https://github.com/user-attachments/assets/b5a94b31-39b4-4a9b-87f5-8fd08d47f3e9" />



## Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing
- Transformer Models
- BERT
- Python
- Streamlit
- Machine Learning
- Data Preprocessing
- Model Evaluation
- Git & GitHub


## Author

Pothurai Rupasree

GitHub:
https://github.com/Rupasree2827

LinkedIn:
https://linkedin.com/in/rupa-pothurai

Portfolio:
https://rupasree-portfolio-tau.vercel.app

