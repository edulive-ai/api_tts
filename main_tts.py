"""
Ví dụ sử dụng module VixTTS không qua API Flask
"""
import module_tts as vixtts
import logging
import os
from pathlib import Path

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo thư mục output trong thư mục hiện tại
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def tts():
    """Ví dụ chuyển đổi một đoạn văn bản dài"""
    # Khởi tạo mô hình nếu chưa được khởi tạo
    if not vixtts.is_initialized():
        logger.info("Đang khởi tạo mô hình TTS...")
        if not vixtts.initialize():
            logger.error("Không thể khởi tạo mô hình TTS")
            return
    
    # Đoạn văn bản dài
    long_text = """
    Trí tuệ nhân tạo là một lĩnh vực của khoa học máy tính tập trung vào việc phát triển các hệ thống thông minh. 
    Các hệ thống này có khả năng thực hiện các nhiệm vụ thông thường đòi hỏi trí thông minh của con người. 
    Một số ứng dụng phổ biến của trí tuệ nhân tạo bao gồm nhận dạng giọng nói, xử lý ngôn ngữ tự nhiên, 
    thị giác máy tính, và học máy. Trong những năm gần đây, trí tuệ nhân tạo đã phát triển nhanh chóng 
    và đã tạo ra những tiến bộ đáng kể trong nhiều lĩnh vực.
    
    Một trong những ứng dụng thú vị của trí tuệ nhân tạo là khả năng chuyển đổi văn bản thành giọng nói. 
    Công nghệ này cho phép máy tính đọc văn bản với giọng điệu tự nhiên, gần giống với giọng nói của con người. 
    Điều này mở ra nhiều khả năng ứng dụng trong giáo dục, hỗ trợ người khuyết tật, và cải thiện trải nghiệm 
    người dùng trong các sản phẩm kỹ thuật số.
    """
    
    try:
        # Chuyển đổi text thành speech
        logger.info("Đang chuyển đổi văn bản dài thành giọng nói...")
        audio_buffer = vixtts.text_to_speech(long_text, lang="vi", normalize=True)
        
        # Lưu giọng nói vào file trong thư mục output
        output_file = OUTPUT_DIR / "output.wav"
        with open(output_file, "wb") as f:
            f.write(audio_buffer.getvalue())
        
        logger.info(f"Đã lưu giọng nói vào file: {output_file}")
        
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        
if __name__ == "__main__":
    logger.info("Bắt đầu chạy các ví dụ sử dụng module VixTTS")
    tts()
    logger.info("success")