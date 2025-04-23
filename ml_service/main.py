# ml_service/main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import implicit
import json
import os
from datetime import datetime
from typing import Dict, Any
import logging
import joblib
from contextlib import asynccontextmanager
from pymongo.errors import PyMongoError
from tenacity import retry, stop_after_attempt, wait_fixed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Environment variables (set in Docker)
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_PATH = os.getenv("DATA_PATH", "./data/spotify_dataset.csv")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "spotify_recsys"

memory = joblib.Memory(MODEL_DIR, verbose=0)

model = None
user_to_idx = {}
track_to_idx = {}
item_user_matrix = None
track_metadata = {}
vectorizer = None
tfidf_matrix = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, user_to_idx, track_to_idx, item_user_matrix, client

    client = MongoClient(MONGO_URI, maxPoolSize = 10, minPoolSize = 3)

    try:
        await load_or_train_model()
    except Exception as e:
        logger.critical(f"Initialization failed: {str(e)}")
        raise
    
    yield
    
    logger.info("Closing resources...")
    global vectorizer, tfidf_matrix
    vectorizer = None
    tfidf_matrix = None
    client.close()
    logger.info("MongoDB connections closed")

app = FastAPI(lifespan=lifespan)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def load_or_train_model():
    global model, user_to_idx, track_to_idx, item_user_matrix
    
    try:
        model = joblib.load(f"{MODEL_DIR}/latest_model.pkl")
        user_to_idx = joblib.load(f"{MODEL_DIR}/user_mapping.pkl")
        track_to_idx = joblib.load(f"{MODEL_DIR}/track_mapping.pkl")
        item_user_matrix = load_npz(f"{MODEL_DIR}/matrix.npz")
        logger.info("Loaded existing model")
    except FileNotFoundError:
        logger.warning("No model found - initial training")
        train_model(use_mongo_data=False)
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise

    initialize_content_features() 
    

def train_model(use_mongo_data: bool = True):
    global model, user_to_idx, track_to_idx, item_user_matrix
    
    logger.info("Starting model training...")
    
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    df = df.dropna(subset=['user_id', 'trackname', 'artistname'])
    df = df.drop_duplicates(subset=['user_id', 'trackname'])

    if use_mongo_data:
        mongo_data = get_mongo_interactions()
        mongo_data = mongo_data[['user_id', 'trackname', 'artistname']]
        df = pd.concat([df, mongo_data], ignore_index=True)
    
    df = df.dropna(subset=['user_id', 'trackname', 'artistname'])
    df = df.drop_duplicates(subset=['user_id', 'trackname'])

    user_to_idx = {user: idx for idx, user in enumerate(df['user_id'].unique())}
    track_to_idx = {track: idx for idx, track in enumerate(df['trackname'].unique())}

    rows = df['user_id'].map(user_to_idx)
    cols = df['trackname'].map(track_to_idx)
    data = np.ones(len(df))
    
    user_item = csr_matrix((data, (rows, cols)), 
                         shape=(len(user_to_idx), len(track_to_idx)))
    item_user_matrix = user_item.T.tocsr()

    if model:
        logger.info("Performing incremental training")
        model.partial_fit(item_user_matrix)
    else:
        logger.info("Training new model")
        model = implicit.als.AlternatingLeastSquares(
            factors=64, 
            iterations=20,
            random_state=42
        )
        model.fit(item_user_matrix)
    
    save_model()
    logger.info("Model training completed successfully")

def save_model():
    """Persist model and mappings to disk"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/latest_model.pkl")
    joblib.dump(user_to_idx, f"{MODEL_DIR}/user_mapping.pkl")
    joblib.dump(track_to_idx, f"{MODEL_DIR}/track_mapping.pkl")
    save_npz(f"{MODEL_DIR}/matrix.npz", item_user_matrix)

def get_mongo_interactions() -> pd.DataFrame:
    """Fetch user interactions from MongoDB"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db.user_interactions
    
    try:
        data = list(collection.find({}, {'_id': 0}))
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching MongoDB data: {str(e)}")
        return pd.DataFrame()

@app.get("/recommend/{user_id}")
async def recommend(user_id: str, n: int = 10) -> Dict[str, Any]:
    """Get recommendations for a user"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    if user_id not in user_to_idx:
        logger.info(f"New user detected: {user_id}")
        return {
            "user_id": user_id,
            "recommendations": get_content_recommendations(n),
            "message": "Popular tracks for new user"
        }
    
    user_idx = user_to_idx[user_id]
    ids, scores = model.recommend(user_idx, item_user_matrix[user_idx], N=n)
    
    recommendations = [{
        "track": list(track_to_idx.keys())[i],
        "score": float(s)
    } for i, s in zip(ids, scores)]
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "model_version": datetime.fromtimestamp(
            os.path.getctime(f"{MODEL_DIR}/latest_model.pkl")
        ).isoformat()
    }

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining with latest data"""
    background_tasks.add_task(train_model, use_mongo_data=True)
    return {"message": "Retraining started in background"}

def get_cold_start_recommendations(n: int = 10) -> list:
    """Get most popular tracks across all users"""
    df = pd.read_csv(DATA_PATH)
    popular_tracks = df['trackname'].value_counts().head(n).index.tolist()
    return [{"track": t, "score": 1.0} for t in popular_tracks]

@memory.cache
def initialize_content_features():
    """Precompute content features"""
    global track_metadata, vectorizer, tfidf_matrix
    
    df = pd.read_csv(DATA_PATH)
    track_metadata = df.groupby('trackname').agg({
        'artistname': 'first',
        'playlistname': lambda x: ' '.join(set(x))
    }).to_dict(orient='index')
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = [
        f"{info['artistname']} {info['playlistname']}" 
        for info in track_metadata.values()
    ]
    tfidf_matrix = vectorizer.fit_transform(corpus)

def get_content_recommendations(n: int = 10) -> list:
    """Recommend based on track popularity + content similarity"""
    # Get popular tracks
    popular = get_cold_start_recommendations(n//2)
    
    # Get content-similar tracks
    random_track = np.random.choice(list(track_metadata.keys()))
    track_idx = list(track_metadata.keys()).index(random_track)
    similarities = cosine_similarity(tfidf_matrix[track_idx], tfidf_matrix)
    similar_indices = np.argsort(similarities[0])[-n//2:][::-1]
    
    content_based = [{
        "track": list(track_metadata.keys())[i],
        "score": float(similarities[0][i])
    } for i in similar_indices]
    
    return popular + content_based

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)