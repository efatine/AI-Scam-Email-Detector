import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer

DATA_FILEPATH = 'emails.csv'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'spam'
SCAM_LABEL_VALUE = 1

MODEL_ARTIFACTS_PATH = 'scam_model_artifacts.joblib'

# Expanded Metadata features
METADATA_FEATURES = [
    'percent_caps', 'percent_punct', 'link_count', 'word_count', 
    'urgency_count', 'money_count', 'generic_greeting'
]

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

def compute_metadata_features(texts):
    features = []
    # Trigger words lists
    urgency_words = ['urgent', 'immediate', 'verify', 'suspended', 'risk', 'action required', 'deadline', 'alert']
    money_words = ['$', 'usd', 'bitcoin', 'bank', 'account', 'credit', 'transfer', 'invoice', 'payment']
    
    for text in texts:
        if not isinstance(text, str):
            text = ""
            
        total_len = len(text)
        text_lower = text.lower()
        
        if total_len == 0:
            features.append([0] * len(METADATA_FEATURES))
            continue
            
        # 1. Structural Features
        caps_count = sum(1 for c in text if c.isupper())
        percent_caps = (caps_count / total_len) * 100
        
        punct_count = sum(1 for c in text if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        percent_punct = (punct_count / total_len) * 100
        
        link_count = len(re.findall(r'http[s]?://|www\.', text))
        word_count = len(text.split())
        
        # 2. Psychological Triggers (The "Smart" Features)
        urgency_count = sum(1 for w in urgency_words if w in text_lower)
        money_count = sum(1 for w in money_words if w in text_lower)
        
        # Scams often use "Dear User" instead of your name
        generic_greeting = 1 if re.search(r'dear (customer|user|client|member|email holder)', text_lower) else 0
        
        features.append([percent_caps, percent_punct, link_count, word_count, urgency_count, money_count, generic_greeting])
        
    return np.array(features)

def load_and_prepare_data():
    try:
        df = pd.read_csv(DATA_FILEPATH)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILEPATH}' was not found.")
        return None, None
    
    # Basic cleanup
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        return None, None
    
    df['is_scam'] = np.where(df[LABEL_COLUMN] == SCAM_LABEL_VALUE, 1, 0)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
    return df[TEXT_COLUMN], df['is_scam']

def train_model():
    X, y = load_and_prepare_data()
    if X is None: return

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split: {len(X_train)} train, {len(X_test)} test.")

    # 1. Vectorize (TF-IDF)
    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 2. Metadata & Psychology
    print("Calculating Metadata & Psychology features...")
    X_train_meta = compute_metadata_features(X_train)
    X_test_meta = compute_metadata_features(X_test)
    
    scaler = StandardScaler()
    X_train_meta_scaled = scaler.fit_transform(X_train_meta)
    X_test_meta_scaled = scaler.transform(X_test_meta)

    # 3. SBERT (Context)
    print("Encoding with SBERT...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    X_train_sbert = sbert_model.encode(X_train.tolist(), show_progress_bar=True)
    X_test_sbert = sbert_model.encode(X_test.tolist(), show_progress_bar=True)

    sbert_scaler = StandardScaler()
    X_train_sbert_scaled = sbert_scaler.fit_transform(X_train_sbert)
    X_test_sbert_scaled = sbert_scaler.transform(X_test_sbert)

    # 4. Train Base Models
    print("Training Base Models...")
    
    # Model A: Lexical (TF-IDF + Meta)
    X_train_lexical = hstack([X_train_tfidf, X_train_meta_scaled]).tocsr()
    model_tfidf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    model_tfidf.fit(X_train_lexical, y_train)
    
    # Model B: Semantic (SBERT)
    model_sbert = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    model_sbert.fit(X_train_sbert_scaled, y_train)

    # 5. Stacking: Train "The Manager"
    print("Training Stacking Ensemble (The Manager)...")
    
    # We need predictions on the training set to train the Manager. 
    # We use cross_val_predict to avoid overfitting (leaking data).
    train_proba_tfidf = cross_val_predict(model_tfidf, X_train_lexical, y_train, cv=3, method='predict_proba')[:, 1]
    train_proba_sbert = cross_val_predict(model_sbert, X_train_sbert_scaled, y_train, cv=3, method='predict_proba')[:, 1]
    
    # The Manager looks at [TFIDF_Prob, SBERT_Prob, Metadata_Features] to decide
    X_train_stack = np.column_stack((train_proba_tfidf, train_proba_sbert, X_train_meta_scaled))
    
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model.fit(X_train_stack, y_train)

    # 6. Evaluation
    print("\n--- Evaluating Stacked Model ---")
    
    # Get test probabilities
    X_test_lexical = hstack([X_test_tfidf, X_test_meta_scaled]).tocsr()
    test_proba_tfidf = model_tfidf.predict_proba(X_test_lexical)[:, 1]
    test_proba_sbert = model_sbert.predict_proba(X_test_sbert_scaled)[:, 1]
    
    # Stack test inputs
    X_test_stack = np.column_stack((test_proba_tfidf, test_proba_sbert, X_test_meta_scaled))
    
    # Final prediction
    final_pred = meta_model.predict(X_test_stack)
    print(f"Accuracy: {accuracy_score(y_test, final_pred)*100:.2f}%")
    print(classification_report(y_test, final_pred))

    # Save everything
    artifacts = {
        "model_tfidf": model_tfidf,
        "model_sbert": model_sbert,
        "meta_model": meta_model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "sbert_scaler": sbert_scaler
    }
    joblib.dump(artifacts, MODEL_ARTIFACTS_PATH)
    print("Training Complete.")

# LIME Explainer Function
def get_lime_explanation(email_text, artifacts):
    # Wrapper to make our complex pipeline look like a simple function for LIME
    def predictor(texts):
        # This function must return shape (n_samples, 2) -> [prob_not_scam, prob_scam]
        results = []
        for t in texts:
            # 1. Features
            t_vec = artifacts['vectorizer'].transform([t])
            t_meta = compute_metadata_features([t])
            t_meta_scaled = artifacts['scaler'].transform(t_meta)
            t_lex = hstack([t_vec, t_meta_scaled]).tocsr()
            
            # 2. SBERT
            # Check if model object exists (API) or needs loading
            sbert = artifacts.get('sbert_model_obj') or SentenceTransformer(SBERT_MODEL_NAME)
            t_sbert = sbert.encode([t])
            t_sbert_scaled = artifacts['sbert_scaler'].transform(t_sbert)
            
            # 3. Base Probs
            p_tfidf = artifacts['model_tfidf'].predict_proba(t_lex)[0][1]
            p_sbert = artifacts['model_sbert'].predict_proba(t_sbert_scaled)[0][1]
            
            # 4. Meta Prob
            stack_in = np.column_stack((p_tfidf, p_sbert, t_meta_scaled))
            final_p_scam = artifacts['meta_model'].predict_proba(stack_in)[0][1]
            
            results.append([1 - final_p_scam, final_p_scam])
            
        return np.array(results)

    explainer = LimeTextExplainer(class_names=['Safe', 'Scam'])
    # num_features=6 returns the top 6 most influential words
    exp = explainer.explain_instance(email_text, predictor, num_features=6, num_samples=200)
    return exp.as_list()

def predict_email(email_text, artifacts=None):
    if artifacts is None:
        try:
            artifacts = joblib.load(MODEL_ARTIFACTS_PATH)
        except:
            return {"error": "Model not found"}

    # 1. Preprocessing
    text_tfidf = artifacts['vectorizer'].transform([email_text])
    text_meta = compute_metadata_features([email_text])
    text_meta_scaled = artifacts['scaler'].transform(text_meta)
    
    # 2. Base Predictions
    text_lexical = hstack([text_tfidf, text_meta_scaled]).tocsr()
    tfidf_prob = artifacts['model_tfidf'].predict_proba(text_lexical)[0][1]

    # Handle SBERT
    sbert = artifacts.get('sbert_model_obj') or SentenceTransformer(SBERT_MODEL_NAME)
    text_sbert = sbert.encode([email_text])
    text_sbert_scaled = artifacts['sbert_scaler'].transform(text_sbert)
    sbert_prob = artifacts['model_sbert'].predict_proba(text_sbert_scaled)[0][1]

    # 3. Meta Prediction (The Manager)
    stack_input = np.column_stack((tfidf_prob, sbert_prob, text_meta_scaled))
    final_prob = artifacts['meta_model'].predict_proba(stack_input)[0][1]
    
    is_scam = final_prob > 0.5

    # 4. Generate Explanation (LIME)
    # We only run LIME if the confidence is reasonably high to save time, or just run it always.
    lime_explanation = get_lime_explanation(email_text, artifacts)

    # Extract meta features for UI display
    meta_raw = text_meta[0] # [caps, punct, link, word, urgency, money, greeting]
    
    result = {
        "is_scam": bool(is_scam),
        "prediction": "Scam" if is_scam else "Safe",
        "confidence_percent": f"{final_prob * 100:.2f}%",
        "breakdown": {
            "tfidf_score": f"{tfidf_prob * 100:.1f}%",
            "sbert_score": f"{sbert_prob * 100:.1f}%"
        },
        "triggers": {
            "urgency_count": int(meta_raw[4]),
            "money_words": int(meta_raw[5]),
            "generic_greeting": bool(meta_raw[6])
        },
        "lime_explanation": lime_explanation 
        # Returns list like [('urgent', 0.15), ('meeting', -0.05)]
    }
    
    return result

if __name__ == "__main__":
    train_model()