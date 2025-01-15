import tensorflow as tf
from transformers import TFVideoMAEModel, VideoMAEFeatureExtractor
from app.config import Config

class VideoFeatureExtractor:
    def __init__(self):
        self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(Config.VIDEO_MODEL_NAME)
        self.model = TFVideoMAEModel.from_pretrained(Config.VIDEO_MODEL_NAME)

    def extract_features(self, video_frames):
        inputs = self.feature_extractor(video_frames, return_tensors="tf")
        outputs = self.model(**inputs)
        return tf.reduce_mean(outputs.last_hidden_state, axis=1)

class VideoProtoNet:
    def __init__(self):
        self.feature_extractor = VideoFeatureExtractor()

    def compute_similarity(self, query_features, support_features):
        normalized_query = tf.nn.l2_normalize(query_features, axis=1)
        normalized_support = tf.nn.l2_normalize(support_features, axis=1)
        return tf.matmul(normalized_query, normalized_support, transpose_b=True) 