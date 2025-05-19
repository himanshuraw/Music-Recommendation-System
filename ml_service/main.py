from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import implicit
import os
from typing import Dict, Any, List
import logging
import logstash
import sys
import joblib
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
import uuid
import time
from fastapi import Request
from bson import ObjectId

# ─── Logging Configuration ───────────────────────────────────────
logger = logging.getLogger("RecommendationServiceLogger")
logger.setLevel(logging.INFO)

# Custom filter to handle missing fields
class OptionalFieldsFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'null'
        if not hasattr(record, 'endpoint'):
            record.endpoint = 'null'
        if not hasattr(record, 'method'):
            record.method = 'null'
        return True

# Logstash Handler
try:
    logstash_handler = logstash.TCPLogstashHandler(
        host='logstash',
        port=5044,
        version=1
    )
    logstash_handler.addFilter(OptionalFieldsFilter())
    logger.addHandler(logstash_handler)
except Exception as e:
    logger.error(f"Failed to initialize Logstash handler: {str(e)}")

# Console Handler with JSON formatting
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
    '"module": "%(module)s", "function": "%(funcName)s", '
    '"message": "%(message)s", "request_id": "%(request_id)s", '
    '"endpoint": "%(endpoint)s", "method": "%(method)s", "service": "recommendation"}'
)
stream_handler.setFormatter(formatter)
stream_handler.addFilter(OptionalFieldsFilter())
logger.addHandler(stream_handler)

# Clear existing handlers for Uvicorn to avoid duplication
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = [stream_handler]
uvicorn_logger.propagate = False

# Environment variables
load_dotenv()
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
        logger.info("Initializing recommendation service", extra={
            'request_id': 'system',
            'endpoint': 'lifespan',
            'method': 'N/A'
        })
        await load_or_train_model()
        logger.info("Service initialization completed successfully", extra={
            'request_id': 'system',
            'endpoint': 'lifespan',
            'method': 'N/A'
        })
    except Exception as e:
        logger.error("Initialization failed", extra={
            'error': str(e),
            'request_id': 'system',
            'endpoint': 'lifespan',
            'method': 'N/A'
        }, exc_info=True)
        raise

    yield

    logger.info("Closing resources", extra={
        'request_id': 'system',
        'endpoint': 'lifespan',
        'method': 'N/A'
    })
    client.close()
    logger.info("MongoDB connections closed", extra={
        'request_id': 'system',
        'endpoint': 'lifespan',
        'method': 'N/A'
    })

app = FastAPI(lifespan=lifespan)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info("Request started", extra={
        'request_id': request_id,
        'endpoint': request.url.path,
        'method': request.method
    })

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error("Request failed", extra={
            'request_id': request_id,
            'error': str(e),
            'endpoint': request.url.path,
            'method': request.method
        }, exc_info=True)
        raise

    process_time = time.time() - start_time
    logger.info("Request completed", extra={
        'request_id': request_id,
        'endpoint': request.url.path,
        'method': request.method,
        'processing_time': process_time,
        'status_code': response.status_code
    })

    return response

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def load_or_train_model():
    global model, user_to_idx, track_to_idx, item_user_matrix, track_metadata
    try:
        model = joblib.load(f"{MODEL_DIR}/latest_model.pkl")
        user_to_idx = joblib.load(f"{MODEL_DIR}/user_mapping.pkl")
        track_to_idx = joblib.load(f"{MODEL_DIR}/track_mapping.pkl")
        item_user_matrix = load_npz(f"{MODEL_DIR}/matrix.npz")
        track_metadata = joblib.load(f"{MODEL_DIR}/track_metadata.pkl")
        logger.info("Model loaded successfully", extra={
            'request_id': 'system',
            'endpoint': 'load_or_train_model',
            'method': 'N/A',
            'model_dir': MODEL_DIR,
            'num_users': len(user_to_idx),
            'num_tracks': len(track_to_idx)
        })
    except FileNotFoundError:
        logger.warning("No model found - initial training", extra={
            'request_id': 'system',
            'endpoint': 'load_or_train_model',
            'method': 'N/A',
            'model_dir': MODEL_DIR
        })
        train_model(use_mongo_data=False)

def train_model(use_mongo_data: bool = True):
    global model, user_to_idx, track_to_idx, item_user_matrix, track_metadata
    logger.info("Starting model training", extra={
        'request_id': 'system',
        'endpoint': 'train_model',
        'method': 'N/A',
        'use_mongo_data': use_mongo_data
    })

    try:
        # Load and prepare data
        df = pd.read_csv(DATA_PATH, dtype={'user_id': str})
        df = df.dropna(subset=['user_id', 'track_id', 'artist_id'])
        df = df.drop_duplicates(subset=['user_id', 'track_id'])

        # Load mappings
        track_map = pd.read_csv("./data/track_map.csv")
        artist_map = pd.read_csv("./data/artist_map.csv")

        # Build track metadata
        track_artist_pairs = df[['track_id', 'artist_id']].drop_duplicates()
        track_metadata_df = track_artist_pairs.merge(
            track_map, on='track_id', how='left'
        ).merge(
            artist_map, on='artist_id', how='left'
        )
        track_metadata_df = track_metadata_df.drop_duplicates(subset='track_id').set_index('track_id')
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
        logger.info("Training new model", extra={
            'request_id': 'system',
            'endpoint': 'train_model',
            'method': 'N/A',
            'num_users': len(user_to_idx),
            'num_tracks': len(track_to_idx),
            'interactions': len(df)
        })
        model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20, random_state=42)
        model.fit(item_user_matrix)

        save_model()
        logger.info("Model training completed successfully", extra={
            'request_id': 'system',
            'endpoint': 'train_model',
            'method': 'N/A'
        })

    except Exception as e:
        logger.error("Model training failed", extra={
            'request_id': 'system',
            'endpoint': 'train_model',
            'method': 'N/A',
            'error': str(e)
        }, exc_info=True)
        raise

def save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/latest_model.pkl")
    joblib.dump(user_to_idx, f"{MODEL_DIR}/user_mapping.pkl")
    joblib.dump(track_to_idx, f"{MODEL_DIR}/track_mapping.pkl")
    joblib.dump(track_metadata, f"{MODEL_DIR}/track_metadata.pkl")
    save_npz(f"{MODEL_DIR}/matrix.npz", item_user_matrix)
    logger.info("Model artifacts saved", extra={
        'request_id': 'system',
        'endpoint': 'save_model',
        'method': 'N/A',
        'model_dir': MODEL_DIR
    })

def get_mongo_interactions() -> pd.DataFrame:
    try:
        with MongoClient(MONGODB_URI) as client:
            db = client[DB_NAME]

            likes = db['likes'].find({}, {'_id': 0, 'userId': 1, 'track_id': 1, 'artist_id': 1})
            likes_df = pd.DataFrame(list(likes)).rename(columns={'userId': 'mongo_user_id'})
            likes_df['mongo_user_id'] = likes_df['mongo_user_id'].apply(
                lambda x: ObjectId(x) if not isinstance(x, ObjectId) else x
            )

            mappings = db['useridmappings'].find({}, {'_id': 0, 'mongo_user_id': 1, 'numerical_user_id': 1})
            mappings_df = pd.DataFrame(list(mappings))

            if mappings_df.empty:
                mappings_df = pd.DataFrame(columns=['mongo_user_id', 'numerical_user_id'])
            else:
                mappings_df['mongo_user_id'] = mappings_df['mongo_user_id'].apply(
                    lambda x: ObjectId(x) if not isinstance(x, ObjectId) else x
                )

            merged_df = pd.merge(likes_df, mappings_df, on='mongo_user_id', how='left')
            missing_users = merged_df[merged_df['numerical_user_id'].isna()]['mongo_user_id'].drop_duplicates()

            if not missing_users.empty:
                logger.info("Creating new user mappings", extra={
                    'request_id': 'system',
                    'endpoint': 'get_mongo_interactions',
                    'method': 'N/A',
                    'num_new_users': len(missing_users)
                })

                transformed_df = pd.read_csv(DATA_PATH, dtype={'user_id': str})
                existing_ids = [int(uid) for uid in transformed_df['user_id'].dropna().unique() if str(uid).isdigit()]
                existing_map_ids = mappings_df['numerical_user_id'].dropna().astype(int).tolist()

                start_id = max(existing_ids + existing_map_ids + [0]) + 1
                new_ids = list(range(start_id, start_id + len(missing_users)))

                new_mappings_df = pd.DataFrame({
                    'mongo_user_id': missing_users.values,
                    'numerical_user_id': new_ids
                })

                db['useridmappings'].insert_many(new_mappings_df.to_dict('records'))

                mappings_df = pd.concat([mappings_df, new_mappings_df], ignore_index=True)

            merged_df = pd.merge(likes_df, mappings_df, on='mongo_user_id', how='left')
            merged_df['user_id'] = merged_df['numerical_user_id'].astype(str)

            logger.info("MongoDB interactions processed", extra={
                'request_id': 'system',
                'endpoint': 'get_mongo_interactions',
                'method': 'N/A',
                'num_interactions': len(merged_df)
            })

            return merged_df[['user_id', 'track_id', 'artist_id']]

    except Exception as e:
        logger.error("MongoDB interaction error", extra={
            'request_id': 'system',
            'endpoint': 'get_mongo_interactions',
            'method': 'N/A',
            'error': str(e)
        }, exc_info=True)
        return pd.DataFrame()

@app.get("/recommend/{user_id}", response_model=Dict[str, Any])
async def recommend(user_id: str, n: int = 10) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    if model is None:
        logger.error("Recommendation request for untrained model", extra={
            'request_id': request_id,
            'endpoint': f'/recommend/{user_id}',
            'method': 'GET'
        })
        raise HTTPException(status_code=503, detail="Model not trained yet")

    response = {"user_id": user_id, "recommendations": []}

    if user_id not in user_to_idx:
        logger.info("New user recommendations", extra={
            'request_id': request_id,
            'endpoint': f'/recommend/{user_id}',
            'method': 'GET',
            'user_id': user_id
        })
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

        logger.info("Recommendations generated", extra={
            'request_id': request_id,
            'endpoint': f'/recommend/{user_id}',
            'method': 'GET',
            'user_id': user_id,
            'num_recommendations': len(recommendations)
        })
        response["recommendations"] = recommendations
        return response

    except Exception as e:
        logger.error("Recommendation error", extra={
            'request_id': request_id,
            'endpoint': f'/recommend/{user_id}',
            'method': 'GET',
            'user_id': user_id,
            'error': str(e)
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating recommendations")

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    logger.info("Retraining triggered", extra={
        'request_id': request_id,
        'endpoint': '/retrain',
        'method': 'POST'
    })
    background_tasks.add_task(train_model, use_mongo_data=True)
    return {"message": "Retraining started in background"}

def get_popular_tracks(n: int = 10) -> List[Dict[str, Any]]:
    request_id = str(uuid.uuid4())
    df = pd.read_csv(DATA_PATH)
    popular_tracks = df['track_id'].value_counts().head(n).index.tolist()
    recommendations = [{
        "track": track_metadata.get(track_id, {}).get("trackname", "Unknown Track"),
        "track_id": int(track_id),
        "artist": track_metadata.get(track_id, {}).get("artistname", "Unknown Artist"),
        "artist_id": int(track_metadata.get(track_id, {}).get("artist_id", -1)),
        "score": 1.0
    } for track_id in popular_tracks]
    logger.info("Popular tracks generated", extra={
        'request_id': request_id,
        'endpoint': 'get_popular_tracks',
        'method': 'N/A',
        'num_tracks': len(recommendations)
    })
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)