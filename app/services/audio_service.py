import librosa
import tensorflow as tf
from app.models.audio_models import AudioProtoNet

class AudioService:
    def __init__(self):
        self.proto_net = AudioProtoNet()
        self.target_sr = 16000

    def _load_audio(self, audio_file):
        audio, sr = librosa.load(audio_file, sr=self.target_sr)
        return audio, sr

    def zero_shot_inference(self, file):
        audio, sr = self._load_audio(file)
        features = self.proto_net.feature_extractor.extract_features(audio, sr)
        return {"features": features.numpy().mean(axis=0).tolist()}

    def one_shot_inference(self, query_file, example_file):
        query_audio, query_sr = self._load_audio(query_file)
        example_audio, example_sr = self._load_audio(example_file)

        query_features = self.proto_net.feature_extractor.extract_features(query_audio, query_sr)
        example_features = self.proto_net.feature_extractor.extract_features(example_audio, example_sr)

        similarity = self.proto_net.compute_similarity(query_features, example_features)
        return {"similarity_score": float(similarity.numpy()[0][0])}

    def few_shot_inference(self, query_file, example_files):
        query_audio, query_sr = self._load_audio(query_file)
        query_features = self.proto_net.feature_extractor.extract_features(query_audio, query_sr)

        example_features_list = []
        for example_file in example_files:
            example_audio, example_sr = self._load_audio(example_file)
            example_features = self.proto_net.feature_extractor.extract_features(
                example_audio, example_sr
            )
            example_features_list.append(example_features)

        example_features = tf.concat(example_features_list, axis=0)
        similarities = self.proto_net.compute_similarity(query_features, example_features)
        return {"similarity_scores": similarities.numpy()[0].tolist()} 