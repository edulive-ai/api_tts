from flask import Flask, request, send_file, jsonify, make_response
import os
import string
import torch
import torchaudio
import numpy as np
from datetime import datetime
from underthesea import sent_tokenize
from unidecode import unidecode
import time
import io
import traceback
import sys
from dotenv import load_dotenv
import logging
import requests
try:
    from vinorm import TTSnorm
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except Exception as e:
    send_telegram_message(f"Lỗi import: {e}")

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

# Cấu hình Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")  # Lấy token từ biến môi trường
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")  # Lấy chat ID từ biến môi trường
ENABLE_TELEGRAM = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)  # Chỉ bật khi có đủ thông tin

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vndoc-crawler")

def send_telegram_message(message):
    """
    Gửi tin nhắn tới Telegram Bot
    
    Args:
        message (str): Nội dung tin nhắn
    """
    if not ENABLE_TELEGRAM:
        logger.debug("Bỏ qua gửi tin nhắn Telegram vì không có cấu hình.")
        return
        
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logger.debug(f"Đã gửi tin nhắn Telegram: {message[:50]}...")
    except Exception as e:
        logger.error(f"Lỗi khi gửi tin nhắn Telegram: {str(e)}")

app = Flask(__name__)

# Biến toàn cục
vixtts_model = None
speaker_embedding = None
gpt_cond_latent = None
DEFAULT_REFERENCE_AUDIO = "./voices/Seren.wav"

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return None
    
    send_telegram_message(f"Đang tải mô hình từ: {xtts_checkpoint}")
    send_telegram_message(f"Config: {xtts_config}")
    send_telegram_message(f"Vocab: {xtts_vocab}")
    
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    send_telegram_message("Đang tải mô hình XTTS!")
    XTTS_MODEL.load_checkpoint(config,
                              checkpoint_path=xtts_checkpoint,
                              vocab_path=xtts_vocab,
                              use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    send_telegram_message("Đã tải xong mô hình!")
    return XTTS_MODEL

def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename

def calculate_keep_len(text, lang):
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = (
        text.count(".")
        + text.count("!")
        + text.count("?")
        + text.count(",")
    )

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1

def normalize_vietnamese_text(text):
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
        send_telegram_message(f"Lỗi chuẩn hoá văn bản: {e}")
        return text

def run_tts(model, lang, tts_text, normalize_text=True):
    """
    Chạy chuyển đổi văn bản thành giọng nói
    """
    global gpt_cond_latent, speaker_embedding
    
    start_time = time.time()
    
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    if normalize_text and lang == "vi":
        try:
            tts_text = normalize_vietnamese_text(tts_text)
            send_telegram_message(f"Văn bản sau khi chuẩn hoá: {tts_text}")
        except Exception as e:
            send_telegram_message(f"Lỗi chuẩn hoá: {e}")

    if lang in ["ja", "zh-cn"]:
        tts_texts = tts_text.split("。")
    else:
        tts_texts = sent_tokenize(tts_text)
        
    send_telegram_message(f"Văn bản được chia thành {len(tts_texts)} đoạn")

    wav_chunks = []
    for i, text in enumerate(tts_texts):
        if text.strip() == "":
            continue
        
        send_telegram_message(f"Đang xử lý đoạn {i+1}/{len(tts_texts)}: '{text}'")

        try:
            wav_chunk = model.inference(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=30,
                top_p=0.85,
            )

            # Kiểm tra và chuyển đổi numpy array sang tensor nếu cần
            if isinstance(wav_chunk["wav"], np.ndarray):
                wav_tensor = torch.tensor(wav_chunk["wav"])
                send_telegram_message(f"Đã chuyển đổi numpy array thành tensor có shape {wav_tensor.shape}")
            else:
                wav_tensor = wav_chunk["wav"]
                send_telegram_message(f"Sử dụng tensor có sẵn với shape {wav_tensor.shape}")

            # Điều chỉnh độ dài cho câu ngắn
            keep_len = calculate_keep_len(text, lang)
            if keep_len > 0 and wav_tensor.shape[0] > keep_len:
                wav_tensor = wav_tensor[:keep_len]
                send_telegram_message(f"Đã cắt tensor còn {wav_tensor.shape[0]} điểm")

            wav_chunks.append(wav_tensor)
            send_telegram_message(f"Đã xử lý xong đoạn {i+1}")
            
        except Exception as e:
            send_telegram_message(f"Lỗi khi xử lý đoạn '{text}': {e}")
            send_telegram_message(traceback.format_exc())

    # Đảm bảo chúng ta có ít nhất một đoạn âm thanh hợp lệ
    if not wav_chunks:
        raise Exception("Không có đoạn âm thanh nào được tạo ra")

    # Ghép các đoạn và lưu
    try:
        send_telegram_message(f"Đang ghép {len(wav_chunks)} đoạn âm thanh")
        send_telegram_message(f"Loại dữ liệu: {type(wav_chunks[0])}")
        
        # Đảm bảo tất cả các đoạn là tensor
        tensor_chunks = []
        for i, chunk in enumerate(wav_chunks):
            if not isinstance(chunk, torch.Tensor):
                tensor_chunks.append(torch.tensor(chunk))
                send_telegram_message(f"Đã chuyển đổi đoạn {i} từ {type(chunk)} sang tensor")
            else:
                tensor_chunks.append(chunk)
        
        # Ghép các đoạn tensor
        out_wav = torch.cat(tensor_chunks, dim=0).unsqueeze(0)
        send_telegram_message(f"Đã ghép thành công, shape kết quả: {out_wav.shape}")
        
        # Tạo buffer để lưu dữ liệu
        buffer = io.BytesIO()
        
        # Lưu tensor vào buffer
        torchaudio.save(buffer, out_wav, 24000, format="wav")
        buffer.seek(0)
        
        send_telegram_message(f"Đã lưu âm thanh vào buffer với kích thước {len(buffer.getvalue())} bytes")
        
        end_time = time.time()
        send_telegram_message(f"Thời gian xử lý TTS: {end_time - start_time:.2f} giây")
        
        return buffer
        
    except Exception as e:
        send_telegram_message(f"Lỗi khi lưu âm thanh: {e}")
        send_telegram_message(traceback.format_exc())
        raise

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Thiếu trường 'text' trong yêu cầu"}), 400
            
        text = data['text']
        lang = data.get('lang', 'vi')
        normalize = data.get('normalize', True)
        
        send_telegram_message(f"Xử lý yêu cầu TTS: '{text}' (ngôn ngữ: {lang}, chuẩn hoá: {normalize})")
        
        # Xử lý TTS
        buffer = run_tts(
            model=vixtts_model,
            lang=lang,
            tts_text=text,
            normalize_text=normalize
        )
        
        # Kiểm tra và trả về dữ liệu âm thanh
        content_length = len(buffer.getvalue())
        send_telegram_message(f"Kích thước dữ liệu âm thanh: {content_length} bytes")
        
        if content_length == 0:
            return jsonify({"error": "File âm thanh được tạo ra rỗng"}), 500
            
        # Tạo response với dữ liệu âm thanh
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=output.wav'
        response.headers['Content-Length'] = str(content_length)
        
        send_telegram_message("Đang trả về response...")
        return response
        
    except Exception as e:
        send_telegram_message(f"Lỗi trong endpoint text_to_speech: {e}")
        send_telegram_message(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# Endpoint để kiểm tra xem server có hoạt động không
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "model_loaded": vixtts_model is not None,
        "embeddings_loaded": speaker_embedding is not None and gpt_cond_latent is not None,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

# Endpoint đơn giản để kiểm tra
@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

def initialize():
    """Khởi tạo mô hình và speaker embeddings khi server bắt đầu"""
    global vixtts_model, gpt_cond_latent, speaker_embedding
    
    send_telegram_message("Đang khởi tạo mô hình TTS...")
    try:
        # Tải mô hình
        vixtts_model = load_model(
            xtts_checkpoint="model/model.pth",
            xtts_config="model/config.json",
            xtts_vocab="model/vocab.json"
        )
        
        if vixtts_model is None:
            send_telegram_message("Không thể tải mô hình")
            return False
            
        # Kiểm tra xem file âm thanh tham chiếu có tồn tại không
        if not os.path.exists(DEFAULT_REFERENCE_AUDIO):
            send_telegram_message(f"Không tìm thấy file âm thanh tham chiếu: {DEFAULT_REFERENCE_AUDIO}")
            return False
            
        # Tính toán trước speaker embeddings
        send_telegram_message(f"Đang tạo speaker embeddings từ {DEFAULT_REFERENCE_AUDIO}...")
        gpt_cond_latent, speaker_embedding = vixtts_model.get_conditioning_latents(
            audio_path=DEFAULT_REFERENCE_AUDIO,
            gpt_cond_len=vixtts_model.config.gpt_cond_len,
            max_ref_length=vixtts_model.config.max_ref_len,
            sound_norm_refs=vixtts_model.config.sound_norm_refs,
        )
        send_telegram_message("Đã tạo xong speaker embeddings!")
        
        # Kiểm tra thiết bị GPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            send_telegram_message(f"Sử dụng GPU: {device_name} với {total_memory:.2f} GB bộ nhớ")
        else:
            send_telegram_message("Không có GPU, sử dụng CPU")
        
        return True
        
    except Exception as e:
        send_telegram_message(f"Lỗi khởi tạo: {e}")
        send_telegram_message(traceback.format_exc())
        return False

if __name__ == '__main__':
    # Khởi tạo mô hình trước khi bắt đầu server
    if initialize():
        # Bắt đầu Flask server
        send_telegram_message("Bắt đầu chạy Flask server...")
        app.run(host='0.0.0.0', port=1010, debug=False)
    else:
        send_telegram_message("Không thể khởi tạo. Thoát.")