import os
from PIL import Image

def get_average_image_size(directory):
    # 檢查目錄是否存在
    if not os.path.exists(directory):
        print(f"目錄 '{directory}' 不存在！")
        return

    # 初始化統計變數
    total_images = 0
    total_width = 0
    total_height = 0

    # 遍歷目錄中的文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 確保是圖片文件
        try:
            with Image.open(file_path) as img:
                total_images += 1
                width, height = img.size
                total_width += width
                total_height += height
        except Exception as e:
            print(f"跳過文件: {filename}（錯誤: {e}）")

    # 計算並輸出平均尺寸
    if total_images > 0:
        avg_width = total_width / total_images
        avg_height = total_height / total_images
        print(f"總圖片數量: {total_images}")
        print(f"平均寬度: {avg_width:.2f}px, 平均高度: {avg_height:.2f}px")
    else:
        print("沒有找到有效的圖片文件。")

# 指定圖片所在目錄
directory_path = "data/my_imgs"
get_average_image_size(directory_path)
