import tensorflow as tf
from transformers import TFWav2Vec2Model, Wav2Vec2Processor
from app.config import Config

class AudioFeatureExtractor:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(Config.AUDIO_MODEL_NAME)
        self.model = TFWav2Vec2Model.from_pretrained(Config.AUDIO_MODEL_NAME)

    def extract_features(self, audio_signal, sample_rate):
        inputs = self.processor(
            audio_signal,
            sampling_rate=sample_rate,
            return_tensors="tf"
        )
        outputs = self.model(**inputs)
        return tf.reduce_mean(outputs.last_hidden_state, axis=1)

class AudioProtoNet:
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()

    def compute_similarity(self, query_features, support_features):
        normalized_query = tf.nn.l2_normalize(query_features, axis=1)
        normalized_support = tf.nn.l2_normalize(support_features, axis=1)
        return tf.matmul(normalized_query, normalized_support, transpose_b=True) 