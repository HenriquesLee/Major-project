import tensorflow as tf
from transformers import TFCLIPModel, CLIPProcessor
from app.config import Config

class CLIPZeroShot:
    def __init__(self):
        self.model = TFCLIPModel.from_pretrained(Config.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)

    def predict(self, image, candidate_labels):
        inputs = self.processor(
            images=image,
            text=candidate_labels,
            return_tensors="tf",
            padding=True
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = tf.nn.softmax(logits_per_image, axis=1)
        
        return {label: float(prob) for label, prob in zip(candidate_labels, probs[0])}

class ProtoNetImage:
    def __init__(self):
        self.model = TFCLIPModel.from_pretrained(Config.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)

    def get_embeddings(self, image):
        inputs = self.processor(images=image, return_tensors="tf")
        return self.model.get_image_features(**inputs)

    def compute_similarity(self, query_embedding, support_embeddings):
        # Compute cosine similarity using TensorFlow
        normalized_query = tf.nn.l2_normalize(query_embedding, axis=1)
        normalized_support = tf.nn.l2_normalize(support_embeddings, axis=1)
        return tf.matmul(normalized_query, normalized_support, transpose_b=True) 