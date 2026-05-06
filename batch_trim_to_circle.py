import cv2
import numpy as np
import pillow_heif
from PIL import Image
import os
import shutil
from datetime import datetime, timedelta

# 註冊 HEIF 開啟器
pillow_heif.register_heif_opener()

def process_images_in_folder(input_folder, output_folder):
    # 建立輸出資料夾與存檔資料夾
    archive_folder = os.path.join(input_folder, "archive")
    for folder in [output_folder, archive_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.heic')

    if not os.path.exists(input_folder):
        print(f"⚠️ 找不到輸入資料夾：{input_folder}")
        return

    # 1. 執行清理作業：刪除 archive 中超過 14 天的檔案
    cleanup_old_archives(archive_folder, days=14)

    files = os.listdir(input_folder)
    processed_count = 0
    
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        
        # 跳過資料夾 (避免處理到 archive 資料夾)
        if os.path.isdir(input_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue
            
        base_name = os.path.splitext(filename)[0]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        new_filename = f"{base_name}_{timestamp}_crop.jpg"
        output_path = os.path.join(output_folder, new_filename)
        
        print(f"🔄 正在處理：{filename} ...")
        
        # 讀取圖片
        img_bgr = None
        try:
            if ext == '.heic':
                img_pil = Image.open(input_path)
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ 讀取失敗 {filename}: {e}")
            continue
            
        if img_bgr is None:
            continue

        # 預處理與霍夫圓變換
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000, 
            param1=50, param2=30, minRadius=800, maxRadius=2000
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            
            y1, y2 = max(0, y - r), min(img_bgr.shape[0], y + r)
            x1, x2 = max(0, x - r), min(img_bgr.shape[1], x + r)
            
            cropped_img = img_bgr[y1:y2, x1:x2]
            
            # 儲存圖片並搬移原始檔
            try:
                is_success, buffer = cv2.imencode(".jpg", cropped_img)
                if is_success:
                    buffer.tofile(output_path)
                    print(f"✅ 成功裁切並儲存至：{new_filename}")
                    processed_count += 1
                    
                    # 2. 搬移至 archive 並加上日期標籤
                    today_str = now.strftime("%Y%m%d")
                    archive_filename = f"{base_name}_archived_{today_str}{ext}"
                    archive_path = os.path.join(archive_folder, archive_filename)
                    shutil.move(input_path, archive_path)
                    print(f"📦 已封存原始圖片至：{archive_filename}")

                else:
                    print(f"❌ 儲存失敗：{new_filename}")
            except Exception as e:
                print(f"❌ 寫入或搬移失敗 {new_filename}: {e}")
        else:
            print(f"⚠️ 無法檢測到明顯的圓盤，跳過此圖片：{filename}")
            
    print(f"🎉 批次處理完成！共成功裁切 {processed_count} 張圖片。")

def cleanup_old_archives(archive_folder, days):
    """檢查 archive 資料夾，刪除檔名中標記超過指定天數的檔案"""
    print("🧹 開始檢查是否需要清理舊封存檔...")
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for filename in os.listdir(archive_folder):
        # 尋找檔名中的日期標記 (例如 _archived_20260506)
        if "_archived_" in filename:
            try:
                date_str = filename.rsplit("_archived_", 1)[1].split(".")[0]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    file_path = os.path.join(archive_folder, filename)
                    os.remove(file_path)
                    print(f"🗑️ 已刪除超過 {days} 天的封存檔：{filename}")
            except ValueError:
                # 如果檔名解析失敗則跳過
                pass

if __name__ == "__main__":
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"
    process_images_in_folder(INPUT_DIR, OUTPUT_DIR)