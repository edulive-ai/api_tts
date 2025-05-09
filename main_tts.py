"""
Chương trình tương tác sử dụng module TTS - Chỉ tải model một lần
"""
import module_tts as vixtts
import logging
import os
from pathlib import Path
import time
from datetime import datetime

# Cấu hình logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tạo thư mục output trong thư mục hiện tại
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def initialize_tts():
    """Khởi tạo model TTS"""
    logger.info("Đang khởi tạo mô hình TTS...")
    start_time = time.time()
    
    if not vixtts.is_initialized():
        if vixtts.initialize():
            elapsed_time = time.time() - start_time
            logger.info(f"Khởi tạo mô hình TTS thành công! (Thời gian: {elapsed_time:.2f} giây)")
            return True
        else:
            logger.error("Không thể khởi tạo mô hình TTS")
            return False
    else:
        logger.info("Mô hình TTS đã được khởi tạo trước đó")
        return True

def convert_text_to_speech(text, lang="vi", normalize=True):
    """Chuyển đổi văn bản thành giọng nói và lưu vào file"""
    if not text.strip():
        logger.warning("Văn bản trống, bỏ qua chuyển đổi")
        return False
    
    try:
        # Tạo tên file dựa trên thời gian hiện tại
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"output_{timestamp}.wav"
        
        # Hiển thị thông tin
        logger.info(f"Đang chuyển đổi văn bản ({len(text)} ký tự)...")
        logger.info(f"Ngôn ngữ: {lang}, Chuẩn hóa: {normalize}")
        
        # Đo thời gian xử lý
        start_time = time.time()
        
        # Chuyển đổi text thành speech
        audio_buffer = vixtts.text_to_speech(text, lang=lang, normalize=normalize)
        
        # Lưu giọng nói vào file
        with open(output_file, "wb") as f:
            f.write(audio_buffer.getvalue())
        
        # Tính thời gian xử lý
        elapsed_time = time.time() - start_time
        file_size_kb = os.path.getsize(output_file) / 1024
        
        logger.info(f"Đã lưu giọng nói vào file: {output_file}")
        logger.info(f"Kích thước file: {file_size_kb:.2f} KB")
        logger.info(f"Thời gian xử lý: {elapsed_time:.2f} giây")
        
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi văn bản: {e}")
        return False

def show_menu():
    """Hiển thị menu chức năng"""
    print("\n" + "="*50)
    print("CHƯƠNG TRÌNH CHUYỂN VĂN BẢN THÀNH GIỌNG NÓI".center(50))
    print("="*50)
    print("1. Nhập văn bản để chuyển đổi thành giọng nói (tiếng Việt)")
    print("2. Đặt lại mô hình (nếu gặp lỗi)")
    print("0. Thoát chương trình")
    print("-"*50)
    choice = input("Vui lòng chọn chức năng (0-2): ")
    return choice

def list_output_files():
    """Hiển thị danh sách các file đã tạo"""
    files = sorted(list(OUTPUT_DIR.glob("*.wav")), key=os.path.getmtime, reverse=True)
    
    if not files:
        print("\nChưa có file nào được tạo.")
        return
    
    print(f"\nDanh sách {len(files)} file đã tạo (gần đây nhất lên đầu):")
    print("-"*80)
    print(f"{'STT':<5}{'Tên file':<30}{'Kích thước':<15}{'Thời gian tạo':<25}")
    print("-"*80)
    
    for i, file in enumerate(files, 1):
        file_stats = os.stat(file)
        size_kb = file_stats.st_size / 1024
        mod_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:<5}{file.name:<30}{f'{size_kb:.2f} KB':<15}{mod_time:<25}")

def main():
    """Hàm chính của chương trình"""
    print("Khởi động chương trình chuyển văn bản thành giọng nói...")
    
    # Khởi tạo model TTS (chỉ một lần duy nhất)
    if not initialize_tts():
        print("Khởi tạo mô hình thất bại. Thoát chương trình.")
        return
    
    # Vòng lặp chính của chương trình
    running = True
    while running:
        choice = show_menu()
        
        if choice == "1":
            # Nhập văn bản tiếng Việt
            print("\nNhập văn bản tiếng Việt (gõ 'END' ở dòng riêng để kết thúc):")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            
            text = "\n".join(lines)
            if text.strip():
                convert_text_to_speech(text, lang="vi", normalize=True)
            else:
                print("Không có văn bản được nhập.")
        
        elif choice == "2":
            # Đặt lại mô hình
            print("\nĐang đặt lại mô hình...")
            if vixtts.reset_tts():
                print("Đặt lại mô hình thành công!")
            else:
                print("Đặt lại mô hình thất bại!")
        
        elif choice == "0":
            # Thoát chương trình
            print("\nĐang thoát chương trình...")
            running = False
        
        else:
            print("\nLựa chọn không hợp lệ. Vui lòng chọn lại.")

if __name__ == "__main__":
    try:
        main()
        print("\nChương trình đã kết thúc.")
    except KeyboardInterrupt:
        print("\nChương trình bị dừng bởi người dùng.")
    except Exception as e:
        logger.error(f"Lỗi không mong đợi: {e}")
        print(f"\nĐã xảy ra lỗi: {e}")