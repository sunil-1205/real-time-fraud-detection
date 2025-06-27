# 🕵️ Real-Time Fraud Detection in Financial Transactions

A real-time machine learning system to detect fraudulent credit card transactions using FastAPI, Kafka, XGBoost, and Streamlit or Power BI. Built for production-ready fraud detection with cloud deployment and streaming support.

---

## 🚀 Project Features

- 🔍 **ML Model**: XGBoost classifier trained on imbalanced financial data
- ⚡ **FastAPI**: Real-time prediction API
- 🔁 **Kafka**: Simulates streaming transaction data
- 📊 **Dashboard**: Live monitoring with Streamlit or Power BI
- ☁️ **AWS Deployment**: Dockerized and deployed on EC2
- 💾 **Data Logging**: Outputs predictions to CSV or database

---

## 📁 Project Structure

real_time_fraud_detection/
│
├── app/ # FastAPI app & Streamlit dashboard
├── data/ # Raw dataset (ignored from git)
├── models/ # Trained model (.pkl)
├── output/ # Real-time logs for Power BI
├── scripts/ # Training, Kafka producer/consumer
├── Dockerfile
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🧠 Model Training

1. Load and preprocess dataset (Kaggle Credit Card Fraud)
2. Handle imbalance with SMOTE
3. Train with RandomForest & XGBoost
4. Evaluate using ROC-AUC, F1, confusion matrix

---

## 🛠️ Tech Stack

| Area        | Tools |
|-------------|-------|
| ML          | XGBoost, scikit-learn, SMOTE |
| API         | FastAPI, Uvicorn |
| Streaming   | Apache Kafka |
| Dashboard   | Streamlit / Power BI |
| Deployment  | Docker, AWS EC2 |

---

## 🧪 How to Run Locally

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python scripts/train_model.py

# 4. Run FastAPI
uvicorn app.main:app --reload

# 5. Run Streamlit
streamlit run app/dashboard.py
