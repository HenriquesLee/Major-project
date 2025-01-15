import tensorflow as tf
from PIL import Image
from app.models.image_models import CLIPZeroShot, ProtoNetImage

class ImageService:
    def __init__(self):
        self.zero_shot_model = CLIPZeroShot()
        self.proto_net = ProtoNetImage()

    def zero_shot_inference(self, file):
        image = Image.open(file).convert('RGB')
        candidate_labels = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
        return self.zero_shot_model.predict(image, candidate_labels)

    def one_shot_inference(self, query_file, example_file):
        query_image = Image.open(query_file).convert('RGB')
        example_image = Image.open(example_file).convert('RGB')

        query_embedding = self.proto_net.get_embeddings(query_image)
        example_embedding = self.proto_net.get_embeddings(example_image)

        similarity = self.proto_net.compute_similarity(query_embedding, example_embedding)
        return {"similarity_score": float(similarity.numpy()[0][0])}

    def few_shot_inference(self, query_file, example_files):
        query_image = Image.open(query_file).convert('RGB')
        example_images = [Image.open(f).convert('RGB') for f in example_files]

        query_embedding = self.proto_net.get_embeddings(query_image)
        example_embeddings = tf.concat([
            self.proto_net.get_embeddings(img) for img in example_images
        ], axis=0)

        similarities = self.proto_net.compute_similarity(query_embedding, example_embeddings)
        return {"similarity_scores": similarities.numpy()[0].tolist()} 