from flask import Flask, request, jsonify, make_response
import os
import string
import torch
import torchaudio
import numpy as np
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from functools import lru_cache
import time
import io
import traceback
from pathlib import Path
import re

# Configure paths and constants
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"
VOICES_DIR = BASE_DIR / "voices"
DEFAULT_REFERENCE_AUDIO = VOICES_DIR / "Seren.wav"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
logger = logging.getLogger("vixtts")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler(LOG_DIR / "vixtts.log", maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize Flask app
app = Flask(__name__)

# Lazy imports to avoid loading until needed
def load_dependencies():
    """Lazy load dependencies to improve startup time"""
    try:
        global TTSnorm, XttsConfig, Xtts, sent_tokenize, unidecode
        from vinorm import TTSnorm
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from underthesea import sent_tokenize
        from unidecode import unidecode
        return True
    except Exception as e:
        logger.error(f"Error importing dependencies: {e}")
        return False

# Global variables for model and embeddings
class TTSModel:
    def __init__(self):
        self.model = None
        self.speaker_embedding = None
        self.gpt_cond_latent = None
        self.is_initialized = False

tts_engine = TTSModel()

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model():
    """Load the XTTS model and return it"""
    clear_gpu_cache()
    
    xtts_checkpoint = MODEL_DIR / "model.pth"
    xtts_config = MODEL_DIR / "config.json"
    xtts_vocab = MODEL_DIR / "vocab.json"
    
    # Check if files exist
    for file_path in [xtts_checkpoint, xtts_config, xtts_vocab]:
        if not file_path.exists():
            logger.error(f"Model file not found: {file_path}")
            return None
    
    logger.info(f"Loading model from: {xtts_checkpoint}")
    
    config = XttsConfig()
    config.load_json(str(xtts_config))
    
    model = Xtts.init_from_config(config)
    logger.info("Loading XTTS model...")
    
    model.load_checkpoint(config,
                         checkpoint_path=str(xtts_checkpoint),
                         vocab_path=str(xtts_vocab),
                         use_deepspeed=False)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"Using GPU: {device_name} with {total_memory:.2f} GB memory")
    else:
        logger.info("No GPU available, using CPU")

    logger.info("Model loaded successfully!")
    return model

@lru_cache(maxsize=128)
def normalize_vietnamese_text(text):
    """Normalize Vietnamese text for better TTS quality with caching for repeated phrases"""
    try:
        text = (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
            .replace("+", "cộng")
            .replace("-", "trừ")
            .replace("*", "nhân")
            .replace("/", "chia")
            .replace("=", "bằng")
        )
        return text
    except Exception as e:
        logger.error(f"Text normalization error: {e}")
        return text

def clean_text(text):
    """Remove non-text characters and standardize spacing"""
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;-()[]\"' \n\tàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ")
    result = ''.join(char if char in allowed_chars else ' ' for char in text)
    return ' '.join(result.split())

def count_words(text):
    """Count words after removing punctuation"""
    for punct in ['.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '"', "'", '...']:
        text = text.replace(punct, ' ')
    words = [word for word in text.split() if word.strip()]
    return len(words)

def calculate_keep_len(text, lang):
    """Calculate how much audio to keep based on text length and language"""
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = sum(text.count(p) for p in [".", "!", "?", ","])

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1

def split_text_into_chunks(text, lang):
    """Split text into appropriate chunks for TTS processing"""
    total_word_count = count_words(text)
    
    # For very short texts, don't split
    if total_word_count < 10:
        logger.info(f"Input text has only {total_word_count} words, keeping as is")
        return [text]
    
    # Split by sentence based on language
    if lang in ["ja", "zh-cn"]:
        raw_sentences = text.split("。")
    else:
        raw_sentences = sent_tokenize(text)
    
    tts_texts = []
    current_chunk = ""
    
    for sentence in raw_sentences:
        if not current_chunk:
            current_chunk = sentence
        else:
            word_count_current = count_words(current_chunk)
            word_count_sentence = count_words(sentence)
            
            if word_count_current >= 10:
                # Check for important keywords in Vietnamese
                important_keywords = ["câu hỏi", "bài tập", "yêu cầu", "đếm", "tính", "giải"]
                is_important = any(keyword in sentence.lower() for keyword in important_keywords)
                
                if word_count_sentence < 10 and is_important:
                    current_chunk += " " + sentence
                else:
                    tts_texts.append(current_chunk)
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
    
    # Add the last chunk if it exists
    if current_chunk and count_words(current_chunk) > 0:
        word_count_last = count_words(current_chunk)
        if word_count_last < 10 and tts_texts:
            tts_texts[-1] += " " + current_chunk
        else:
            tts_texts.append(current_chunk)
    
    # Clean up punctuation at chunk boundaries
    for i in range(len(tts_texts)):
        text = tts_texts[i].strip()
        # Replace sentence-ending punctuation with commas
        text = re.sub(r'[.!?:;]', ',', text)
        # Remove trailing punctuation
        text = text.rstrip(',.')
        tts_texts[i] = text.strip()
    
    logger.info(f"Text split into {len(tts_texts)} chunks")
    for i, text in enumerate(tts_texts):
        logger.info(f"Chunk {i+1}: {count_words(text)} words - '{text}'")
    
    return tts_texts

def run_tts(text, lang="vi", normalize_text=True):
    """Run TTS conversion with proper chunking and optimization"""
    start_time = time.time()
    
    # Normalize text if needed
    if normalize_text and lang == "vi":
        try:
            text = normalize_vietnamese_text(text)
            logger.info(f"Normalized text: {text}")
        except Exception as e:
            logger.error(f"Normalization error: {e}")
    
    # Clean text to remove unwanted characters
    text = clean_text(text)
    logger.info(f"Cleaned text: {text}")
    
    # Split text into appropriate chunks
    chunks = split_text_into_chunks(text, lang)
    
    # Process each chunk
    wav_chunks = []
    for i, chunk_text in enumerate(chunks):
        if not chunk_text.strip():
            continue
        
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: '{chunk_text}'")
        
        try:
            # Generate audio for this chunk
            wav_chunk = tts_engine.model.inference(
                text=chunk_text,
                language=lang,
                gpt_cond_latent=tts_engine.gpt_cond_latent,
                speaker_embedding=tts_engine.speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=30,
                top_p=0.85,
            )
            
            # Convert to tensor if needed
            if isinstance(wav_chunk["wav"], np.ndarray):
                wav_tensor = torch.tensor(wav_chunk["wav"])
            else:
                wav_tensor = wav_chunk["wav"]
            
            # Trim if needed
            keep_len = calculate_keep_len(chunk_text, lang)
            if keep_len > 0 and wav_tensor.shape[0] > keep_len:
                wav_tensor = wav_tensor[:keep_len]
                logger.debug(f"Trimmed tensor to {wav_tensor.shape[0]} points")
            
            wav_chunks.append(wav_tensor)
            logger.info(f"Processed chunk {i+1}")
        except Exception as e:
            logger.error(f"Error processing chunk '{chunk_text}': {e}")
            logger.error(traceback.format_exc())
    
    if not wav_chunks:
        raise Exception("No audio chunks were generated")
    
    try:
        # Combine all chunks
        logger.info(f"Combining {len(wav_chunks)} audio chunks")
        tensor_chunks = [torch.tensor(chunk) if not isinstance(chunk, torch.Tensor) else chunk for chunk in wav_chunks]
        out_wav = torch.cat(tensor_chunks, dim=0).unsqueeze(0)
        
        # Save to buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, out_wav, 24000, format="wav")
        buffer.seek(0)
        
        logger.info(f"Audio saved to buffer, size: {len(buffer.getvalue())} bytes")
        logger.info(f"Total TTS processing time: {time.time() - start_time:.2f} seconds")
        
        return buffer
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        logger.error(traceback.format_exc())
        raise

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
        
        # Check if model is initialized
        if not tts_engine.is_initialized:
            return jsonify({"error": "TTS engine not initialized"}), 503
        
        # Generate audio
        buffer = run_tts(text=text, lang=lang, normalize_text=normalize)
        
        # Validate output
        content_length = len(buffer.getvalue())
        if content_length == 0:
            return jsonify({"error": "Generated audio file is empty"}), 500
        
        # Create response
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=output.wav'
        response.headers['Content-Length'] = str(content_length)
        
        logger.info("Returning response...")
        return response
        
    except Exception as e:
        logger.error(f"Error in text_to_speech endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def initialize():
    """Initialize the TTS model and speaker embeddings"""
    try:
        # Load dependencies
        if not load_dependencies():
            logger.error("Failed to load dependencies")
            return False
        
        # Load model
        tts_engine.model = load_model()
        if tts_engine.model is None:
            logger.error("Failed to load model")
            return False
        
        # Check reference audio
        if not DEFAULT_REFERENCE_AUDIO.exists():
            logger.error(f"Reference audio file not found: {DEFAULT_REFERENCE_AUDIO}")
            return False
        
        # Generate speaker embeddings
        logger.info(f"Generating speaker embeddings from {DEFAULT_REFERENCE_AUDIO}...")
        tts_engine.gpt_cond_latent, tts_engine.speaker_embedding = tts_engine.model.get_conditioning_latents(
            audio_path=str(DEFAULT_REFERENCE_AUDIO),
            gpt_cond_len=tts_engine.model.config.gpt_cond_len,
            max_ref_length=tts_engine.model.config.max_ref_len,
            sound_norm_refs=tts_engine.model.config.sound_norm_refs,
        )
        logger.info("Speaker embeddings generated successfully!")
        
        tts_engine.is_initialized = True
        return True
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    if tts_engine.is_initialized:
        return jsonify({"status": "ok", "gpu": torch.cuda.is_available()}), 200
    else:
        return jsonify({"status": "not_initialized"}), 503

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting TTS server")
    logger.info(f"Log file: {LOG_DIR / 'vixtts.log'}")
    logger.info("=" * 50)

    if initialize():
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=9321, threaded=True)
    else:
        logger.error("Initialization failed. Exiting.")