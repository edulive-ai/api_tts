"""
Main application that uses the Vietnamese Text-to-Speech module
"""
from flask import Flask, request, jsonify, make_response
import module_tts as vixtts
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    if vixtts.is_initialized():
        return jsonify({"status": "ok", "gpu": vixtts.get_gpu_status()}), 200
    else:
        return jsonify({"status": "not_initialized"}), 503

@app.route('/reset', methods=['POST'])
def reset_model():
    """API endpoint to reset the TTS model in case of errors"""
    try:
        logger.info("Resetting TTS model...")
        
        if vixtts.reset_tts():
            logger.info("TTS model reset successfully")
            return jsonify({"status": "success", "message": "TTS model reset successfully"}), 200
        else:
            logger.error("Failed to reset TTS model")
            return jsonify({"status": "error", "message": "Failed to reset TTS model"}), 500
    except Exception as e:
        logger.error(f"Error resetting TTS model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """API endpoint for text-to-speech conversion"""
    try:
        # Validate request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        # Extract parameters
        text = data['text']
        lang = data.get('lang', 'vi')
        normalize = data.get('normalize', True)
        
        logger.info(f"Processing TTS request: '{text[:50]}...' (language: {lang}, normalize: {normalize})")
        
        # Generate audio using the TTS module
        buffer = vixtts.text_to_speech(text, lang, normalize)
        
        # Create response
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=output.wav'
        response.headers['Content-Length'] = str(len(buffer.getvalue()))
        
        logger.info("Returning response...")
        return response
        
    except Exception as e:
        logger.error(f"Error in text_to_speech endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting TTS server")
    logger.info("=" * 50)
    
    # Initialize the TTS module on server startup
    if not vixtts.is_initialized():
        logger.info("Initializing TTS module...")
        if vixtts.initialize():
            logger.info("TTS module initialized successfully")
        else:
            logger.error("Failed to initialize TTS module")
    
    # Start the Flask server
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=9321, threaded=True)