import cv2
import numpy as np
import pillow_heif
from PIL import Image
import os
from datetime import datetime

# 註冊 HEIF 開啟器，讓 PIL 能直接支援讀取 HEIC 格式
pillow_heif.register_heif_opener()

def process_images_in_folder(input_folder, output_folder):
    # 1. 確保輸出資料夾存在，若無則自動建立
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已建立輸出資料夾：{output_folder}")

    # 定義支援的圖片副檔名 (轉小寫比對)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.heic')

    # 檢查輸入資料夾是否存在
    if not os.path.exists(input_folder):
        print(f"⚠️ 找不到輸入資料夾：{input_folder}，請先建立並放入圖片。")
        return

    # 取得資料夾內所有檔案
    files = os.listdir(input_folder)
    processed_count = 0
    
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        # 略過不支援的檔案格式
        if ext not in supported_extensions:
            continue
            
        input_path = os.path.join(input_folder, filename)
        
        # 2. 產生新檔名：舊檔名_{timestamp}_crop.jpg
        base_name = os.path.splitext(filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_filename = f"{base_name}_{timestamp}_crop.jpg"
        output_path = os.path.join(output_folder, new_filename)
        
        print(f"🔄 正在處理：{filename} ...")
        
        # 3. 讀取圖片 (使用特殊寫法以完美支援中文路徑)
        img_bgr = None
        try:
            if ext == '.heic':
                # 處理 HEIC
                img_pil = Image.open(input_path)
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                # 處理 JPG, PNG
                img_bgr = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ 讀取失敗 {filename}: {e}")
            continue
            
        if img_bgr is None:
            print(f"❌ 無法解析圖片內容：{filename}")
            continue

        # 4. 預處理：轉灰度與高斯模糊
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 5. 霍夫圓變換檢測圓形
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=1000, 
            param1=50, 
            param2=30, 
            minRadius=800, # 可依據圖片比例調整
            maxRadius=2000
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            
            # 計算裁切範圍並確保不超出圖片邊界
            y1 = max(0, y - r)
            y2 = min(img_bgr.shape[0], y + r)
            x1 = max(0, x - r)
            x2 = min(img_bgr.shape[1], x + r)
            
            # 執行裁切
            cropped_img = img_bgr[y1:y2, x1:x2]
            
            # 6. 儲存為 JPG (使用 cv2.imencode 以支援中文路徑)
            try:
                is_success, buffer = cv2.imencode(".jpg", cropped_img)
                if is_success:
                    buffer.tofile(output_path)
                    print(f"✅ 成功裁切並儲存至：{new_filename}")
                    processed_count += 1
                else:
                    print(f"❌ 儲存失敗：{new_filename}")
            except Exception as e:
                print(f"❌ 寫入檔案失敗 {new_filename}: {e}")
        else:
            print(f"⚠️ 無法檢測到明顯的圓盤，跳過此圖片：{filename}")
            
    print(f"🎉 批次處理完成！共成功裁切 {processed_count} 張圖片。")

if __name__ == "__main__":
    # 在這裡設定你的資料夾名稱
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"
    
    process_images_in_folder(INPUT_DIR, OUTPUT_DIR)