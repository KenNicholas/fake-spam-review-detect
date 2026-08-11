from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
import contractions
import nltk
import pickle
from nltk.tokenize import word_tokenize
from io import StringIO
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_dict = {
    "fake_review": {"lr": None, "rf": None, "xgb": None, "lstm": None, "tfidf": None, "tokenizer": None},
    "spam": {"lr": None, "rf": None, "xgb": None, "lstm": None, "tfidf": None, "tokenizer": None}
}

def load_pkl(path):
    try: return joblib.load(path)
    except: return None

def load_keras(path):
    try: return load_model(path)
    except: return None

def load_tokenizer(path):
    try:
        with open(path, 'rb') as f: return pickle.load(f)
    except: return None

print("Loading Models... Mohon tunggu.")

models_dict["fake_review"]["tfidf"] = load_pkl("models/klasik/tfidf_vectorizer.pkl")
models_dict["fake_review"]["lr"] = load_pkl("models/klasik/logistic_regression_model.pkl")
models_dict["fake_review"]["rf"] = load_pkl("models/klasik/random_forest_model.pkl")
models_dict["fake_review"]["xgb"] = load_pkl("models/klasik/xgboost_model.pkl")
models_dict["fake_review"]["tokenizer"] = load_tokenizer("models/deep_learning/tokenizer.pkl")
models_dict["fake_review"]["lstm"] = load_keras("models/deep_learning/lstm_word2vec_model.h5")

models_dict["spam"]["tfidf"] = load_pkl("models/klasik_spam/spam_tfidf_vectorizer.pkl")
models_dict["spam"]["lr"] = load_pkl("models/klasik_spam/spam_logistic_regression_model.pkl")
models_dict["spam"]["rf"] = load_pkl("models/klasik_spam/spam_random_forest_model.pkl")
models_dict["spam"]["xgb"] = load_pkl("models/klasik_spam/spam_xgboost_model.pkl")
models_dict["spam"]["tokenizer"] = load_tokenizer("models/deep_learning_spam/spam_tokenizer.pkl")
models_dict["spam"]["lstm"] = load_keras("models/deep_learning_spam/spam_lstm_word2vec_model.h5")

print("Selesai loading model!")

def preprocessing(statement):
    if not isinstance(statement, str): return ""
    statement = statement.lower()
    statement = contractions.fix(statement)
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    tokens = word_tokenize(statement)
    tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
    return ' '.join(tokens)

def analyze_text(text: str, mode: str, model_type: str):
    if mode not in ["fake_review", "spam"]: return {"error": "Mode invalid."}
    
    clean_text = preprocessing(text)
    score = 0.0
    
    if model_type == "lstm":
        tokenizer = models_dict[mode]["tokenizer"]
        lstm_model = models_dict[mode]["lstm"]
        
        if not tokenizer or not lstm_model:
            return {"error": f"File model LSTM untuk {mode} tidak ditemukan."}
            
        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
        prediction = lstm_model.predict(padded, verbose=0)[0][0]
        score = float(prediction) * 100
        
    elif model_type in ["lr", "rf", "xgb"]:
        tfidf = models_dict[mode]["tfidf"]
        clf_model = models_dict[mode][model_type]
        
        if not tfidf or not clf_model:
            return {"error": f"File model {model_type} untuk {mode} tidak ditemukan."}
            
        vec_text = tfidf.transform([clean_text])
        proba = clf_model.predict_proba(vec_text)[0]
        score = float(proba[1]) * 100
    
    else:
        return {"error": "Model type tidak dikenali."}

    is_flagged = score > 50
    confidence = score if is_flagged else 100 - score
    
    if mode == "spam":
        label = "Spam" if is_flagged else "Safe (Ham)"
        reason = "The text contains word order patterns similar to those found in promotional messages or scams." if is_flagged else "The sentence appears similar to ordinary conversational message."
    else:
        label = "Fake" if is_flagged else "Genuine"
        reason = "The writing style bears a strong resemblance to paid review templates or no clear context." if is_flagged else "The review has a natural variety of vocabulary and recounts the experience in detail."
        
    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "is_flagged": is_flagged,
        "reason": reason
    }

class TextRequest(BaseModel):
    text: str
    mode: str
    model_type: str
    
    model_config = {
        "protected_namespaces": ()
    }

@app.post("/api/analyze/text")
def analyze_single(req: TextRequest):
    result = analyze_text(req.text, req.mode, req.model_type)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"status": "success", "result": result}

@app.post("/api/analyze/file")
async def analyze_file(mode: str = Form(...), model_type: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Hanya mendukung file .csv dan .txt")
    
    content = await file.read()
    decoded = content.decode('utf-8')
    results = []
    
    if file.filename.endswith('.txt'):
        lines = [line.strip() for line in decoded.split('\n') if line.strip()]
        for line in lines:
            res = analyze_text(line, mode, model_type)
            if "error" not in res: results.append(res)
            
    elif file.filename.endswith('.csv'):
        df = pd.read_csv(StringIO(decoded))
        text_col = next((col for col in ['text', 'review', 'v2', 'text_'] if col in df.columns), df.columns[0])
            
        for text in df[text_col].dropna().astype(str):
            res = analyze_text(text, mode, model_type)
            if "error" not in res: results.append(res)
            
    return {"status": "success", "total": len(results), "results": results}