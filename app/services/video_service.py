import cv2
import numpy as np
import tensorflow as tf
from app.models.video_models import VideoProtoNet

class VideoService:
    def __init__(self):
        self.proto_net = VideoProtoNet()

    def _extract_frames(self, video_file, num_frames=16):
        frames = []
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        return frames

    def zero_shot_inference(self, file):
        frames = self._extract_frames(file)
        features = self.proto_net.feature_extractor.extract_features(frames)
        return {"features": features.numpy().mean(axis=0).tolist()}

    def one_shot_inference(self, query_file, example_file):
        query_frames = self._extract_frames(query_file)
        example_frames = self._extract_frames(example_file)

        query_features = self.proto_net.feature_extractor.extract_features(query_frames)
        example_features = self.proto_net.feature_extractor.extract_features(example_frames)

        similarity = self.proto_net.compute_similarity(query_features, example_features)
        return {"similarity_score": float(similarity.numpy()[0][0])}

    def few_shot_inference(self, query_file, example_files):
        query_frames = self._extract_frames(query_file)
        query_features = self.proto_net.feature_extractor.extract_features(query_frames)

        example_features_list = []
        for example_file in example_files:
            example_frames = self._extract_frames(example_file)
            example_features = self.proto_net.feature_extractor.extract_features(example_frames)
            example_features_list.append(example_features)

        example_features = tf.concat(example_features_list, axis=0)
        similarities = self.proto_net.compute_similarity(query_features, example_features)
        return {"similarity_scores": similarities.numpy()[0].tolist()} 