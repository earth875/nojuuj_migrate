import cv2
import numpy as np
import pillow_heif
from PIL import Image
import os

def crop_to_circle_bounding_box(image_path, output_path):
    # 1. 讀取 HEIC 檔案並轉換為 OpenCV 支援的 BGR 格式
    heif_file = pillow_heif.read_heif(image_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data, 
        "raw"
    )
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 2. 預處理：轉灰度與高斯模糊
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 3. 霍夫圓變換檢測圓形 (參數可能需要根據實際圖片微調)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=1000, 
        param1=50, 
        param2=30, 
        minRadius=800, # 假設圓盤佔據大部分畫面，設定較大的最小半徑
        maxRadius=2000
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 取檢測到的第一個（通常是最明顯的）圓形
        x, y, r = circles[0]
        
        # 4. 計算裁切範圍並確保不超出圖片邊界
        y1 = max(0, y - r)
        y2 = min(img_bgr.shape[0], y + r)
        x1 = max(0, x - r)
        x2 = min(img_bgr.shape[1], x + r)
        
        # 5. 執行裁切並儲存為 JPG
        cropped_img = img_bgr[y1:y2, x1:x2]
        cv2.imwrite(output_path, cropped_img)
        print(f"✅ 成功裁切並儲存至：{output_path}")
    else:
        print("⚠️ 無法檢測到明顯的圓盤，請嘗試調整 HoughCircles 參數。")

if __name__ == "__main__":
    input_file = "input.heic"
    output_file = "cropped_plate.jpg"
    
    if os.path.exists(input_file):
        crop_to_circle_bounding_box(input_file, output_file)
    else:
        print(f"找不到檔案 {input_file}，請確認檔案路徑。")