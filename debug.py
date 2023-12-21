# import os
# import cv2
#
# # 设置文件夹路径
# labelcol_dir = "labelcol"
# MRI_Int_dir = "MRI_Int"
#
# # 获取文件夹中的文件列表
# labelcol_files = os.listdir(labelcol_dir)
# MRI_Int_files = os.listdir(MRI_Int_dir)
#
# # 初始化计数器
# cnt = 0
# cnt_lesion = 0
#
# # 遍历文件列表
# for filename in labelcol_files:
#     # 构建图像文件路径
#     labelcol_path = os.path.join(labelcol_dir, filename)
#     MRI_Int_path = os.path.join(MRI_Int_dir, filename)
#
#     # 读取图像
#     labelcol_img = cv2.imread(labelcol_path)
#     MRI_Int_img = cv2.imread(MRI_Int_path)
#
#     # 检查条件并计算分类准确率
#     if ((MRI_Int_img.max(axis=(0, 1)) == 0) and (labelcol_img.max(axis=(0, 1)) == 0)) or \
#             ((MRI_Int_img.max(axis=(0, 1)) > 0) and (labelcol_img.max(axis=(0, 1)) > 0)):
#         cnt_lesion += 1
#
#     cnt += 1
#
# # 计算分类准确率
# accuracy = cnt_lesion / cnt
# print("分类准确率: {:.2f}".format(accuracy))

import os
import cv2

# 定义文件夹路径
labelcol_folder = 'labelcol'
img_folder = 'img'
output_folder_0 = '0'
output_folder_1 = '1'

# 创建文件夹0和文件夹1（如果不存在）
os.makedirs(output_folder_0, exist_ok=True)
os.makedirs(output_folder_1, exist_ok=True)

# 遍历labelcol文件夹中的所有文件
for filename in os.listdir(labelcol_folder):
    # 构建文件路径
    file_path = os.path.join(labelcol_folder, filename)

    # 读取图像
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # 判断图像所有通道的数值是否都为0
    if (image == 0).all():
        # 构建输出路径
        output_path = os.path.join(output_folder_0, filename)

        # 保存图像到文件夹0
        cv2.imwrite(output_path, image)

# 遍历文件夹0下的文件名
for filename in os.listdir(output_folder_0):
    # 构建文件路径
    file_path = os.path.join(output_folder_0, filename)

    # 检查文件名是否与img文件夹中的图片名字相同
    if filename in os.listdir(img_folder):
        # 构建img文件夹中对应图片的路径
        img_file_path = os.path.join(img_folder, filename)

        # 读取img文件夹中对应的图片
        img_image = cv2.imread(img_file_path)

        # 构建输出路径
        output_path = os.path.join(output_folder_1, filename)

        # 保存图片到文件夹1
        cv2.imwrite(output_path, img_image)