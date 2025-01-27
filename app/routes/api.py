from flask import Blueprint, request, jsonify
from app.services.image_service import ImageService
# from app.services.video_service import VideoService
from app.services.audio_service import AudioService

api_bp = Blueprint('api', __name__)

@api_bp.route('/predict/<domain>/<mode>', methods=['POST'])
def predict(domain, mode):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        if domain == 'image':
            service = ImageService()
        elif domain == 'video':
            service = VideoService()
        elif domain == 'audio':
            service = AudioService()
        else:
            return jsonify({'error': 'Invalid domain'}), 400

        if mode == 'zero-shot':
            result = service.zero_shot_inference(file)
        elif mode == 'one-shot':
            example = request.files.get('example')
            if not example:
                return jsonify({'error': 'Example file required for one-shot learning'}), 400
            result = service.one_shot_inference(file, example)
        elif mode == 'few-shot':
            examples = request.files.getlist('examples')
            if len(examples) < 2:
                return jsonify({'error': 'Multiple examples required for few-shot learning'}), 400
            result = service.few_shot_inference(file, examples)
        else:
            return jsonify({'error': 'Invalid mode'}), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500 