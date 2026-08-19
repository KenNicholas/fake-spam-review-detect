# 🛡️ AI Text Sentinel: NLP Analyzer for Fake Review & Spam Detection
AI Text Sentinel is an end-to-end web application designed to detect fraudulent reviews and promotional spam in real time. Powered by dual-engine analytics leveraging both Deep Learning (LSTM with Word2Vec embeddings) and Classic Machine Learning models (Logistic Regression, Random Forest, XGBoost), this system provides rapid classifications, transparent confidence scores, and actionable reasoning to simplify digital content moderation.

## 🔗 Quick Links & Resources
Live Web Application: https://fake-spam-review-detect.vercel.app

GitHub Repository: https://github.com/KenNicholas/fake-spam-review-detect

Backend Model & API Space: [Hugging Face Space](https://huggingface.co/spaces/Ken2707/Fake_Review_Analyzer/tree/main)

Video Demonstration: https://drive.google.com/file/d/1Asi-F5Bk_DiqLH5NGQkEhpFCUAb3ToKT/view?usp=sharing

## 📊 Dataset
This project utilizes two robust, publicly available datasets to train its dual-engine architecture independently:
**1. Fake Review Engine Dataset**

* Source: [Kaggle - 🚨 Fake Reviews Dataset by mexwell](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

* Size: 40,432 reviews (Perfectly balanced 50:50 split).

* Classes: OR (Original Reviews / Authentic) vs CG (Computer-Generated / Fake).

**2. Spam Detection Engine Dataset**

* Source:[Kaggle - SMS Spam Collection Dataset by UCI ML](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

* Size: 5,574 raw text messages.

* Classes: ham (Legitimate) vs spam (Malicious/Promotional).

Classes:

* Fake Review Engine: Genuine vs Fake

* Spam Engine: Ham (Normal) vs Spam

* Preprocessing: Text data underwent rigorous cleaning, including regex filtering, lowercasing, handling contractions, and stop-word removal using NLTK.

While the primary Deep Learning engine utilizes an LSTM model (achieving 98.30% accuracy on spam and 93.71% on fake reviews) due to its superior sequential text processing, I have also trained and deployed classic machine learning alternatives (Logistic Regression, Random Forest, and XGBoost) utilizing TF-IDF vectorization. Users can seamlessly switch and compare these models directly in the web application.

Web application screenshots:

<img width="300" height="200" alt="Screenshot 2026-08-19 191207" src="https://github.com/user-attachments/assets/cba78660-fa59-4720-86b9-e7b3cbb20884" />
<img width="300" height="200" alt="Screenshot 2026-08-19 191241" src="https://github.com/user-attachments/assets/d353fa50-a9ff-49db-ad47-6efe961628ff" />
<img width="300" height="200" alt="Screenshot 2026-08-19 191302" src="https://github.com/user-attachments/assets/49017d3f-b864-4def-914f-5b32e03c3376" />
<img width="300" height="200" alt="Screenshot 2026-08-19 191322" src="https://github.com/user-attachments/assets/d47d7cf0-c4be-4818-bbb5-e542a49f27f5" />


## Key Features
* **Dual-Engine Diagnostics:** Simultaneous real-time analysis to detect both deceptive reviews and promotional spam.

* **Flexible Model Selection:** Compare predictions using various algorithms (LSTM, Logistic Regression, Random Forest, XGBoost).

* **Batch Analysis:** Streamlines workflows with .csv and .txt file upload support to analyze large text datasets all at once.

* **Explainable Insights:** Provides transparent confidence scores and specific reasoning behind every AI prediction.

## Tech Stack
* **Frontend:** React, Vite, Tailwind CSS, Axios, Lucide React

* **Backend:** Python, FastAPI, TensorFlow/Keras, Scikit-Learn, XGBoost, NLTK

* **Cloud & Deployment:** Hugging Face Spaces (Docker Engine), Vercel

---

## How to Run Locally

### 1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/AI-Text-Sentinel.git

cd AI-Text-Sentinel

### 2. Backend Setup (FastAPI & ML Models)
Open a terminal and navigate to the backend folder:

cd backend
# Create a virtual environment (recommended)
python -m venv venv

### On Windows:
venv\Scripts\activate  

### On Mac/Linux:
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload

The backend API will be available at http://localhost:8000

### 3. Open a new terminal and navigate to the frontend folder:
cd frontend
# Install Node modules
npm install

# Start the development server
npm run dev
The web app will be available at http://localhost:5173
