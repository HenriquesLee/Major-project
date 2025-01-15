import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Model configurations
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    AUDIO_MODEL_NAME = "facebook/wav2vec2-base-960h"
    VIDEO_MODEL_NAME = "facebook/timesformer-base-finetuned-k400" 