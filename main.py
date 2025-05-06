from flask import Flask, request, send_file, jsonify, make_response
import os
import string
import torch
import torchaudio
import numpy as np
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from underthesea import sent_tokenize
from unidecode import unidecode
import time
import io
import traceback
import sys
import re
try:
    from vinorm import TTSnorm
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except Exception as e:
    logger.error(f"Lỗi import: {e}")

# Thiết lập logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "vixtts.log")

# Tạo logger
logger = logging.getLogger("vixtts")
logger.setLevel(logging.DEBUG)

# Định dạng log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Log ra file với rotating (giới hạn kích thước)
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Log ra console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Thêm handlers vào logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
    
    logger.info(f"Đang tải mô hình từ: {xtts_checkpoint}")
    logger.info(f"Config: {xtts_config}")
    logger.info(f"Vocab: {xtts_vocab}")
    
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    logger.info("Đang tải mô hình XTTS!")
    XTTS_MODEL.load_checkpoint(config,
                              checkpoint_path=xtts_checkpoint,
                              vocab_path=xtts_vocab,
                              use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    logger.info("Đã tải xong mô hình!")
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
            # .replace('.', ",")
            # .replace('?', ",")
            # .replace('!', ",")
            # .replace('...', ",")
            # .replace(':', ",")
        )
        return text
    except Exception as e:
        logger.error(f"Lỗi chuẩn hoá văn bản: {e}")
        return text

def run_tts(model, lang, tts_text, normalize_text=True):
    """
    Chạy chuyển đổi văn bản thành giọng nói với điều kiện mỗi đoạn phải có ít nhất 10 từ
    """
    global gpt_cond_latent, speaker_embedding
    
    start_time = time.time()
    
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    if normalize_text and lang == "vi":
        try:
            tts_text = normalize_vietnamese_text(tts_text)
            logger.info(f"Văn bản sau khi chuẩn hoá: {tts_text}")
        except Exception as e:
            logger.error(f"Lỗi chuẩn hoá: {e}")

    # Kiểm tra toàn bộ input có ít hơn 10 từ không
    total_word_count = len(tts_text.split())
    if total_word_count < 10:
        logger.info(f"Văn bản đầu vào chỉ có {total_word_count} từ, giữ nguyên không chia")
        tts_texts = [tts_text]
    else:
        if lang in ["ja", "zh-cn"]:
            raw_sentences = tts_text.split("。")
        else:
            # Chia văn bản thành các câu
            raw_sentences = sent_tokenize(tts_text)
            
        # Khởi tạo danh sách để lưu các đoạn đã xử lý
        tts_texts = []
        current_chunk = ""
        
        for i, sentence in enumerate(raw_sentences):
            # Nếu không có đoạn hiện tại, bắt đầu với câu hiện tại
            if not current_chunk:
                current_chunk = sentence
            else:
                # Kiểm tra xem đây có phải là câu cuối cùng không
                is_last_sentence = (i == len(raw_sentences) - 1)
                
                # Đếm số từ trong đoạn hiện tại và câu hiện tại
                word_count_current = len(current_chunk.split())
                word_count_sentence = len(sentence.split())
                word_count_combined = word_count_current + word_count_sentence
                
                # Nếu là câu cuối cùng, xử lý đặc biệt để đảm bảo không còn đoạn nào dưới 10 từ
                if is_last_sentence:
                    # Nếu ghép lại vẫn dưới 10 từ hoặc đoạn hiện tại đã dưới 10 từ, luôn ghép
                    if word_count_combined < 10 or word_count_current < 10:
                        current_chunk += " " + sentence
                        tts_texts.append(current_chunk)
                    else:
                        # Nếu đoạn hiện tại và câu cuối đều đủ 10 từ trở lên, tách riêng
                        if word_count_current >= 10 and word_count_sentence >= 10:
                            tts_texts.append(current_chunk)
                            tts_texts.append(sentence)
                        else:
                            # Đoạn hiện tại đủ 10 từ, nhưng câu cuối không đủ, ghép vào đoạn cuối cùng
                            tts_texts.append(current_chunk)
                            
                            # Ghép câu cuối vào đoạn trước đó nếu có
                            if tts_texts:
                                tts_texts[-1] += " " + sentence
                            else:
                                tts_texts.append(sentence)
                else:
                    # Nếu đoạn hiện tại đã có ít nhất 10 từ, lưu và bắt đầu đoạn mới
                    if word_count_current >= 10:
                        tts_texts.append(current_chunk)
                        current_chunk = sentence
                    else:
                        # Nếu đoạn hiện tại có ít hơn 10 từ, ghép với câu tiếp theo
                        current_chunk += " " + sentence
        
        # Kiểm tra nếu còn đoạn chưa được thêm vào
        if current_chunk and current_chunk not in tts_texts:
            # Nếu đoạn cuối ít hơn 10 từ và có đoạn trước đó, ghép vào đoạn cuối cùng
            word_count_last = len(current_chunk.split())
            if word_count_last < 10 and tts_texts:
                tts_texts[-1] += " " + current_chunk
            else:
                tts_texts.append(current_chunk)
        
    logger.info(f"Văn bản được chia thành {len(tts_texts)} đoạn")

    # Thay thế dấu chấm ở cuối câu bằng dấu phẩy
    for i in range(len(tts_texts)):
    # Cắt khoảng trắng ở hai đầu
        text = tts_texts[i].strip()
        
        # Thay thế tất cả các dấu câu thành dấu phẩy
        text = text.replace('.', ',')
        text = text.replace('!', ',')
        text = text.replace('?', ',')
        text = text.replace(':', ',')
        text = text.replace(';', ',')
        
        # Cập nhật lại đoạn
        tts_texts[i] = text

    # In thông tin về số từ trong mỗi đoạn
    for i, text in enumerate(tts_texts):
        word_count = len(text.split())
        logger.info(f"Đoạn {i+1}: {word_count} từ")

    wav_chunks = []
    for i, text in enumerate(tts_texts):
        if text.strip() == "":
            continue
        
        logger.info(f"Đang xử lý đoạn {i+1}/{len(tts_texts)}: '{text}'")

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
                logger.debug(f"Đã chuyển đổi numpy array thành tensor có shape {wav_tensor.shape}")
            else:
                wav_tensor = wav_chunk["wav"]
                logger.debug(f"Sử dụng tensor có sẵn với shape {wav_tensor.shape}")

            # Điều chỉnh độ dài cho câu ngắn
            keep_len = calculate_keep_len(text, lang)
            if keep_len > 0 and wav_tensor.shape[0] > keep_len:
                wav_tensor = wav_tensor[:keep_len]
                logger.debug(f"Đã cắt tensor còn {wav_tensor.shape[0]} điểm")

            wav_chunks.append(wav_tensor)
            logger.info(f"Đã xử lý xong đoạn {i+1}")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý đoạn '{text}': {e}")
            logger.error(traceback.format_exc())

    # Đảm bảo chúng ta có ít nhất một đoạn âm thanh hợp lệ
    if not wav_chunks:
        raise Exception("Không có đoạn âm thanh nào được tạo ra")

    # Ghép các đoạn và lưu
    try:
        logger.info(f"Đang ghép {len(wav_chunks)} đoạn âm thanh")
        logger.debug(f"Loại dữ liệu: {type(wav_chunks[0])}")
        
        # Đảm bảo tất cả các đoạn là tensor
        tensor_chunks = []
        for i, chunk in enumerate(wav_chunks):
            if not isinstance(chunk, torch.Tensor):
                tensor_chunks.append(torch.tensor(chunk))
                logger.debug(f"Đã chuyển đổi đoạn {i} từ {type(chunk)} sang tensor")
            else:
                tensor_chunks.append(chunk)
        
        # Ghép các đoạn tensor
        out_wav = torch.cat(tensor_chunks, dim=0).unsqueeze(0)
        logger.info(f"Đã ghép thành công, shape kết quả: {out_wav.shape}")
        
        # Tạo buffer để lưu dữ liệu
        buffer = io.BytesIO()
        
        # Lưu tensor vào buffer
        torchaudio.save(buffer, out_wav, 24000, format="wav")
        buffer.seek(0)
        
        logger.info(f"Đã lưu âm thanh vào buffer với kích thước {len(buffer.getvalue())} bytes")
        
        end_time = time.time()
        logger.info(f"Thời gian xử lý TTS: {end_time - start_time:.2f} giây")
        
        return buffer
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu âm thanh: {e}")
        logger.error(traceback.format_exc())
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
        
        logger.info(f"Xử lý yêu cầu TTS: '{text}' (ngôn ngữ: {lang}, chuẩn hoá: {normalize})")
        
        # Xử lý TTS
        buffer = run_tts(
            model=vixtts_model,
            lang=lang,
            tts_text=text,
            normalize_text=normalize
        )
        
        # Kiểm tra và trả về dữ liệu âm thanh
        content_length = len(buffer.getvalue())
        logger.info(f"Kích thước dữ liệu âm thanh: {content_length} bytes")
        
        if content_length == 0:
            return jsonify({"error": "File âm thanh được tạo ra rỗng"}), 500
            
        # Tạo response với dữ liệu âm thanh
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=output.wav'
        response.headers['Content-Length'] = str(content_length)
        
        logger.info("Đang trả về response...")
        return response
        
    except Exception as e:
        logger.error(f"Lỗi trong endpoint text_to_speech: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

def initialize():
    """Khởi tạo mô hình và speaker embeddings khi server bắt đầu"""
    global vixtts_model, gpt_cond_latent, speaker_embedding
    
    logger.info("Đang khởi tạo mô hình TTS...")
    try:
        # Tải mô hình
        vixtts_model = load_model(
            xtts_checkpoint="model/model.pth",
            xtts_config="model/config.json",
            xtts_vocab="model/vocab.json"
        )
        
        if vixtts_model is None:
            logger.error("Không thể tải mô hình")
            return False
            
        # Kiểm tra xem file âm thanh tham chiếu có tồn tại không
        if not os.path.exists(DEFAULT_REFERENCE_AUDIO):
            logger.error(f"Không tìm thấy file âm thanh tham chiếu: {DEFAULT_REFERENCE_AUDIO}")
            return False
            
        # Tính toán trước speaker embeddings
        logger.info(f"Đang tạo speaker embeddings từ {DEFAULT_REFERENCE_AUDIO}...")
        gpt_cond_latent, speaker_embedding = vixtts_model.get_conditioning_latents(
            audio_path=DEFAULT_REFERENCE_AUDIO,
            gpt_cond_len=vixtts_model.config.gpt_cond_len,
            max_ref_length=vixtts_model.config.max_ref_len,
            sound_norm_refs=vixtts_model.config.sound_norm_refs,
        )
        logger.info("Đã tạo xong speaker embeddings!")
        
        # Kiểm tra thiết bị GPU
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"Sử dụng GPU: {device_name} với {total_memory:.2f} GB bộ nhớ")
        else:
            logger.info("Không có GPU, sử dụng CPU")
        
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khởi tạo: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    # Log thông báo khởi động
    logger.info("=" * 50)
    logger.info("Bắt đầu khởi động TTS server")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 50)
    
    # Khởi tạo mô hình trước khi bắt đầu server
    if initialize():
        # Bắt đầu Flask server
        logger.info("Bắt đầu chạy Flask server...")
        app.run(host='0.0.0.0', port=9321, debug=False)
    else:
        logger.error("Không thể khởi tạo. Thoát.")