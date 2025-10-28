import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa ảnh .ppm của bạn
SRC_IMG_DIR = "../data/STARE/images/"
# Nơi lưu các tệp mask (nên là cùng thư mục)
DST_MASK_DIR = "../data/STARE/images/"
# Ngưỡng pixel để phân biệt FOV với nền đen
# Bạn có thể cần điều chỉnh giá trị này (vd: 10, 15, 20)
THRESHOLD_VALUE = 15
# ------------------

def create_fov_mask(image_path, save_path, threshold):
    """
    Tạo và lưu FOV mask cho một ảnh.
    """
    try:
        # 1. Đọc ảnh (OpenCV đọc theo BGR)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Không thể đọc ảnh {image_path}")
            return

        # 2. Tách kênh màu xanh lá (Green channel)
        # Kênh Green (chỉ số 1 trong BGR) thường có độ tương phản tốt nhất
        gray = img[:, :, 1]

        # 3. Áp dụng ngưỡng (threshold)
        # Bất kỳ pixel nào > threshold sẽ thành 255 (trắng), còn lại là 0 (đen)
        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 4. Tìm các đường viền (contours)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Warning: Không tìm thấy contour nào trong {image_path}")
            return

        # 5. Tìm đường viền lớn nhất (theo diện tích)
        largest_contour = max(contours, key=cv2.contourArea)

        # 6. Tạo một mask đen mới
        mask = np.zeros(gray.shape, dtype=np.uint8)

        # 7. Vẽ (tô đầy) đường viền lớn nhất lên mask
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        # 8. Lưu tệp mask
        cv2.imwrite(save_path, mask)

    except Exception as e:
        print(f"Lỗi khi xử lý {image_path}: {e}")

def main():
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(DST_MASK_DIR):
        os.makedirs(DST_MASK_DIR)

    # Tìm tất cả các tệp .ppm trong thư mục nguồn
    image_paths = glob.glob(os.path.join(SRC_IMG_DIR, "*.ppm"))

    if not image_paths:
        print(f"Không tìm thấy tệp .ppm nào trong {SRC_IMG_DIR}")
        print("Hãy kiểm tra lại đường dẫn SRC_IMG_DIR trong script.")
        return

    print(f"Đang tạo FOV masks cho {len(image_paths)} ảnh...")

    # Sử dụng tqdm để xem thanh tiến trình
    for img_path in tqdm(image_paths, desc="Tạo masks"):
        # Lấy tên tệp cơ sở (vd: im0001)
        base_name = os.path.basename(img_path)
        base_name = os.path.splitext(base_name)[0]

        # Tạo đường dẫn lưu tệp mask (vd: data/STARE/images/im0001_mask.png)
        mask_save_path = os.path.join(DST_MASK_DIR, base_name + "_mask.png")

        create_fov_mask(img_path, mask_save_path, THRESHOLD_VALUE)

    print("\nHoàn tất! Tất cả các tệp mask đã được tạo.")

if __name__ == "__main__":
    main()
