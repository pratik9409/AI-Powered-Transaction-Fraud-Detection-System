
# 🧠 AI-Powered Transaction Fraud Detection System

A real-time intelligent fraud detection system that leverages machine learning, anomaly detection, SHAP explainability, and graph neural networks to identify and explain fraudulent financial transactions in banking and fintech environments.

![Screenshot 2025-04-07 164814](https://github.com/user-attachments/assets/412afc0e-5407-4532-be23-ae2f2374101c)


---

## 🚀 Key Features

- ✅ **Real-Time Fraud Detection** with **XGBoost** and **Isolation Forest**
- 🔄 **Streaming Data Pipeline** using **Apache Kafka**
- 🕸️ **Graph Neural Networks** (GNN) for fraud ring detection
- 📊 **Interactive Dashboard** (Flask + Bootstrap + Chart.js)
- 📈 **SHAP Explainability**: Visualize feature contributions
- 💡 **User-Level Stats** and Behavioral Analytics
- 🔒 Risk Scoring & Transaction Flagging
- 🧪 Sample dataset + support for live Kafka transactions

---


---

## 🛠️ Tech Stack

| Layer              | Tools / Libraries                    |
|-------------------|--------------------------------------|
| **Backend**        | Flask, Pandas, NumPy, Joblib         |
| **ML Models**      | XGBoost, Isolation Forest, SHAP      |
| **Streaming**      | Apache Kafka                         |
| **GNN**            | PyTorch Geometric, NetworkX          |
| **Frontend**       | HTML, Bootstrap, Chart.js, JS        |
| **Visualization**  | SHAP, Chart.js                       |

---

## 📦 Setup Instructions

### 1️⃣ Clone the repo

`bash
git clone https://github.com/pratik9409/AI-Powered-Transaction-Fraud-Detection-System.git
cd AI-Powered-Transaction-Fraud-Detection-System
`


2️⃣ Install dependencies
`bash
pip install -r requirements.txt
`

3️⃣ Start Kafka (Optional for Real-Time Stream)
You must have Apache Kafka and Zookeeper installed locally or via Docker.

`bash
zookeeper-server-start.sh config/zookeeper.properties
kafka-server-start.sh config/server.properties
`

4️⃣ Run the backend
`bash
python app.py
`

5️⃣ Open in browser
Visit: http://localhost:5000

🧪 Simulate Transactions
Send transactions via the Kafka producer:

bash
`python kafka_producer.py
`

📉 Example API Payload
POST /api/analyze

`json
{
  "TransactionAmount": 650.75,
  "TransactionDuration": 9.2,
  "LoginAttempts": 1,
  "AccountBalance": 4500.0,
  "TransactionDate": "2024-04-01 11:32:00",
  "PreviousTransactionDate": "2024-03-31 10:00:00",
  "TransactionType": "Debit",
  "Location": "New York",
  "DeviceID": "DVC9012",
  "MerchantID": "MRCH7890",
  "Channel": "Online",
  "CustomerOccupation": "Engineer",
  "AccountID": "AC00128"
}
`

## 🔬 GNN for Fraud Ring Detection
The fraud_gnn/gnn_model.py file loads a transaction graph and applies Graph Neural Networks using PyTorch Geometric to detect suspicious communities and indirect relationships.

## 📌 Roadmap
 Real-time prediction via Kafka

 SHAP explanations in dashboard

 GNN-based fraud ring detection

 Neo4j integration for fraud graph visualization

 Multitenancy support for multiple banks

📄 License
MIT License. See LICENSE for more info.

🙌 Contributions
Pull requests are welcome! Please open an issue first to discuss changes.

👨‍💻 Author
Pratik – @pratik9409

⭐ If you found this project helpful, please star it on GitHub!

---

### Let me know if you'd like:

- A version tailored for **deployment on AWS/GCP**,
- An additional **demo GIF or animated walkthrough**,
- A more technical readme focused on **ML architecture and SHAP explainability**.

