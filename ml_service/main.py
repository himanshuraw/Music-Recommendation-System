from fastapi import FastAPI, BackgroundTasks, HTTPException
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import implicit
import os
from typing import Dict, Any, List
import logging
import joblib
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment variables
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DATA_PATH = os.getenv("DATA_PATH", "./data/transformed_data.csv")
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "test"

# Global model state
model = None
user_to_idx = {}
track_to_idx = {}
item_user_matrix = None
track_metadata = {}
client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, user_to_idx, track_to_idx, item_user_matrix, client, track_metadata
    client = MongoClient(MONGODB_URI, maxPoolSize=10, minPoolSize=3)

    try:
        await load_or_train_model()
    except Exception as e:
        logger.critical(f"Initialization failed: {str(e)}")
        raise
    
    yield
    
    logger.info("Closing resources...")
    client.close()
    logger.info("MongoDB connections closed")

app = FastAPI(lifespan=lifespan)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def load_or_train_model():
    global model, user_to_idx, track_to_idx, item_user_matrix, track_metadata
    try:
        model = joblib.load(f"{MODEL_DIR}/latest_model.pkl")
        user_to_idx = joblib.load(f"{MODEL_DIR}/user_mapping.pkl")
        track_to_idx = joblib.load(f"{MODEL_DIR}/track_mapping.pkl")
        item_user_matrix = load_npz(f"{MODEL_DIR}/matrix.npz")
        track_metadata = joblib.load(f"{MODEL_DIR}/track_metadata.pkl")
        logger.info("Loaded existing model")
    except FileNotFoundError:
        logger.warning("No model found - initial training")
        train_model(use_mongo_data=False)

def train_model(use_mongo_data: bool = True):
    global model, user_to_idx, track_to_idx, item_user_matrix, track_metadata
    logger.info("Starting model training...")
    
    # Load and prepare data
    df = pd.read_csv(DATA_PATH, dtype={'user_id': str})
    df = df.dropna(subset=['user_id', 'track_id', 'artist_id'])
    df = df.drop_duplicates(subset=['user_id', 'track_id'])

    # Load mappings
    track_map = pd.read_csv("./data/track_map.csv")
    artist_map = pd.read_csv("./data/artist_map.csv")

    # Get track-artist relationships from interactions
    track_artist_pairs = df[['track_id', 'artist_id']].drop_duplicates()

    # Build track metadata
    track_metadata_df = track_artist_pairs.merge(
        track_map, on='track_id', how='left'
    ).merge(
        artist_map, on='artist_id', how='left'
    )

    track_metadata_df = track_metadata_df.drop_duplicates(subset='track_id')
    track_metadata_df = track_metadata_df.set_index('track_id')

    track_metadata = track_metadata_df.fillna("Unknown").to_dict(orient='index')

    # Incorporate MongoDB data
    if use_mongo_data:
        mongo_df = get_mongo_interactions()
        df = pd.concat([df, mongo_df], ignore_index=True)

    df = df.dropna(subset=['user_id', 'track_id', 'artist_id'])
    df = df.drop_duplicates(subset=['user_id', 'track_id'])


    # Create mappings
    user_to_idx = {user: idx for idx, user in enumerate(df['user_id'].unique())}
    track_to_idx = {track: idx for idx, track in enumerate(df['track_id'].unique())}

    # Build interaction matrix
    rows = df['user_id'].map(user_to_idx)
    cols = df['track_id'].map(track_to_idx)
    data = np.ones(len(df))
    
    user_item = csr_matrix((data, (rows, cols)), 
                         shape=(len(user_to_idx), len(track_to_idx)))
    item_user_matrix = user_item.T.tocsr()

    # Train/update model
    logger.info("Training new model")
    model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20, random_state=42)
    model.fit(item_user_matrix)
    
    save_model()
    logger.info("Model training completed successfully")

def save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/latest_model.pkl")
    joblib.dump(user_to_idx, f"{MODEL_DIR}/user_mapping.pkl")
    joblib.dump(track_to_idx, f"{MODEL_DIR}/track_mapping.pkl")
    joblib.dump(track_metadata, f"{MODEL_DIR}/track_metadata.pkl")
    save_npz(f"{MODEL_DIR}/matrix.npz", item_user_matrix)

def get_mongo_interactions() -> pd.DataFrame:
    """Fetch likes from MongoDB and map ObjectIds to numerical user_ids"""
    try:
        with MongoClient(MONGODB_URI) as client:
            db = client[DB_NAME]
            logger.info(f"Collections in DB '{DB_NAME}': {db.list_collection_names()}")

            likes = db['likes'].find({}, {'_id': 0, 'userId': 1, 'track_id': 1, 'artist_id': 1})
            likes_df = pd.DataFrame(list(likes))

            if likes_df.empty:
                return pd.DataFrame()

            likes_df.rename(columns={'userId': 'mongo_user_id'}, inplace=True)

            mappings = db['useridmappings'].find({}, {'_id': 0, 'mongo_user_id': 1, 'numerical_user_id': 1})
            mappings_df = pd.DataFrame(list(mappings))

            if mappings_df.empty:
                mappings_df = pd.DataFrame(columns=['mongo_user_id', 'numerical_user_id'])

            merged_df = pd.merge(likes_df, mappings_df, on='mongo_user_id', how='left')

            missing_users = merged_df[merged_df['numerical_user_id'].isna()]['mongo_user_id'].drop_duplicates()

            if not missing_users.empty:
                transformed_df = pd.read_csv(DATA_PATH, dtype={'user_id': str})
                existing_ids = transformed_df['user_id'].dropna().unique().tolist()
                existing_ids = [int(uid) for uid in existing_ids if str(uid).isdigit()]
                max_existing_id = max(existing_ids) if existing_ids else 0

                existing_map_ids = mappings_df['numerical_user_id'].dropna().astype(int).tolist()
                max_map_id = max(existing_map_ids) if existing_map_ids else 0

                start_id = max(max_existing_id, max_map_id) + 1
                new_ids = list(range(start_id, start_id + len(missing_users)))

                new_mappings_df = pd.DataFrame({
                    'mongo_user_id': missing_users.values,
                    'numerical_user_id': new_ids
                })

                db['useridmappings'].insert_many(new_mappings_df.to_dict('records'))

                mappings_df = pd.concat([mappings_df, new_mappings_df], ignore_index=True)
                merged_df = pd.merge(likes_df, mappings_df, on='mongo_user_id', how='left')

            merged_df['user_id'] = merged_df['numerical_user_id'].astype(str)
            merged_df['playlist_id'] = np.nan

            return merged_df[['user_id', 'track_id', 'artist_id', 'playlist_id']]

    except Exception as e:
        logger.error(f"Error fetching MongoDB data: {str(e)}")
        return pd.DataFrame()

@app.get("/recommend/{user_id}", response_model=Dict[str, Any])
async def recommend(user_id: str, n: int = 10) -> Dict[str, Any]:
    """Get recommendations for a user"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    response = {
        "user_id": user_id,
        "recommendations": []
    }

    if user_id not in user_to_idx:
        response["recommendations"] = get_popular_tracks(n)
        response["message"] = "Popular tracks for new user"
        return response

    try:
        user_idx = user_to_idx[user_id]
        ids, scores = model.recommend(user_idx, item_user_matrix[user_idx], N=n)
        recommendations = []
        
        for i, score in zip(ids, scores):
            track_id = list(track_to_idx.keys())[i]
            metadata = track_metadata.get(track_id, {})
            
            recommendations.append({
                "track": metadata.get("trackname", "Unknown Track"),
                "track_id": int(track_id),
                "artist": metadata.get("artistname", "Unknown Artist"),
                "artist_id": int(metadata.get("artist_id", -1)),
                "score": float(score)
            })
            
        response["recommendations"] = recommendations
        return response
    
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining with latest data"""
    background_tasks.add_task(train_model, use_mongo_data=True)
    return {"message": "Retraining started in background"}

def get_popular_tracks(n: int = 10) -> List[Dict[str, Any]]:
    """Get most popular tracks across all users"""
    df = pd.read_csv(DATA_PATH)
    popular_tracks = df['track_id'].value_counts().head(n).index.tolist()
    
    recommendations = []
    for track_id in popular_tracks:
        metadata = track_metadata.get(track_id, {})
        recommendations.append({
            "track": metadata.get("trackname", "Unknown Track"),
            "track_id": int(track_id),
            "artist": metadata.get("artistname", "Unknown Artist"),
            "artist_id": int(metadata.get("artist_id", -1)),
            "score": 1.0
        })
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)