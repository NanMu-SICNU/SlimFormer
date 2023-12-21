# 导入常用库
import SimpleITK as sitk
import PIL.Image as Image
import os
import numpy as np
from datashape import json

import config.ini
import cv2
from pathlib2 import Path


def binaryMask(src, save_test):
    """
        功能：
            - 将mask图像处理成二值化图像
        参数：
            - src：mask的文件位置
            - save_test：处理后的图片存放位置
        返回值：
            - 输出处理后的图像
    """

    os.makedirs(save_test, exist_ok=True)

    for name in os.listdir(src):
        mask = cv2.imread(os.path.join(src, name), cv2.IMREAD_GRAYSCALE)
        ret, mask_binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)  # 如果像素值大于0 则为255；反之则为0
        cv2.imwrite(os.path.join(save_test, name), mask_binary)


def cutImg(src_path, lungmask_path, mask_path, save_path):
    """
        功能：
            - 通过lungmask图像,裁剪掉src图像和mask图像多余部分
        参数：
            - src_path: data数据集
            - mask_path: label数据集（二值化mask图像 groundtruth）
            - save_path: 处理后文件的保存路径
        返回：
            - 将裁减后的图片输出到指定路径
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.makedirs(os.path.join(save_path, 'data'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'label'), exist_ok=True)

    for name in os.listdir(src_path):
        img = cv2.imread(os.path.join(src_path, name), cv2.IMREAD_GRAYSCALE)
        print('裁剪前img的大小：{}'.format(img.shape), type(img))

        lungMask = cv2.imread(os.path.join(lungmask_path, name), cv2.IMREAD_GRAYSCALE)
        print('裁剪前lung_mask的大小：{}'.format(lungMask.shape), type(lungMask))

        mask = cv2.imread(os.path.join(mask_path, name), cv2.IMREAD_GRAYSCALE)
        print('裁剪前mask的大小：{}'.format(mask.shape), type(mask))

        x0, x1, y0, y1 = 0, lungMask.shape[0], 0, lungMask.shape[1]
        for i in range(0, lungMask.shape[0]):
            if (all(lungMask[i] == 0)):
                continue
            else:
                # print(i, mask[i])
                x0 = i
                break

        for i in range(lungMask.shape[0] - 1, 0, -1):
            if (all(lungMask[i] == 0)):
                continue
            else:
                # print(i, mask[i])
                x1 = i
                break

        for i in range(0, lungMask.shape[1]):
            if (all(lungMask[:, i] == 0)):
                continue
            else:
                # print(i, mask[i, :])
                y0 = i
                break

        for i in range(lungMask.shape[1] - 1, 0, -1):
            if (all(lungMask[:, i] == 0)):
                continue
            else:
                # print(i, mask[i, :])
                y1 = i
                break

        croppedImg = img[x0 - 2:x1 + 2, y0 - 2:y1 + 2]  # +1-是为了让图片多留一点点空间
        croppedMask = mask[x0 - 2:x1 + 2, y0 - 2:y1 + 2]
        cv2.imwrite(os.path.join(os.path.join(save_path, 'data'), name), croppedImg)
        cv2.imwrite(os.path.join(os.path.join(save_path, 'label'), name), croppedMask)


def convertPNG(img_path, save_path):
    """
        功能：
            - 将jpg图像转换为png图像
        参数:
            - img_path: jpg文件所在路径
            - save_path: 要保存的文件路径
        返回：
            - 输出处理后的png图像到指定位置
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for name in os.listdir(img_path):
        img1 = Image.open(os.path.join(img_path, name))

        img = img1.convert('RGBA')
        r, g, b, a = img.split()
        a0 = np.array(b)  # 转换为np矩阵
        a1 = cv2.threshold(a0, 10, 255, cv2.THRESH_BINARY)  # 设定阈值
        a2 = Image.fromarray(a1[1])  # 转换为Image的tube格式，注意为a1[1]
        a3 = np.array(a2)
        a4 = Image.fromarray(a3.astype('uint8'))  # 由float16转换为uint8
        img = Image.merge("RGBA", (b, g, r, a4))

        img.save(os.path.join(save_path, name.replace('jpg', 'png')))


def convertJPG(img_path, save_path):
    """
        功能：
            - 将png图像转换为jpg图像
        参数:
            - img_path: png文件所在路径
            - save_path: 要保存的文件路径
        返回：
            - 输出处理后的jpg图像到指定位置
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for name in os.listdir(img_path):
        print(os.path.join(img_path, name))
        img1 = Image.open(os.path.join(img_path, name))

        img = img1.convert('RGB')

        img.save(os.path.join(save_path, name.replace('png', 'jpg')), quality=95)


def renamePic(srcImgPath, save_path):
    """
        功能：
            - 对图片进行重新编号
        参数：
            - srcImgPath: 待编号文件路径
            - save_path: 想要保存的路径
        返回：
            - 将编号后的图片输出保存到指定位置
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    srcImgDir = Path(srcImgPath)

    i = 1
    for item in srcImgDir.rglob("*.jpg"):
        # 获取图片名
        # print(item)
        imgName = item.name
        newName = str(i) + ".jpg"
        # print(imgName, newName)
        i = i + 1
        # 重命名
        print(f"prepare to rename {imgName}")
        item.rename(os.path.join(save_path, newName))


def resize(img_path, save_path):
    """
        功能：
            - 将图像resize成指定尺寸
        参数：
            - img_path: 待处理的文件夹路径
            - save_path: resize后的图像保存的地方
        返回：
            - 输出resize成指定尺寸的图像到指定路径

    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for name in os.listdir(img_path):
        img = Image.open(os.path.join(img_path, name))
        resized = img.resize((128, 128))
        resized = resized.convert('RGB')
        resized.save(os.path.join(save_path, name))


def Img(img_path, save_path):
    """
        功能：
            - 选择img_path中的部分图像写入save_path
        参数：
            img_path：待处理的图片目录
            save_path: 指定的存储目录
        返回：
            img_path处理后的图片输入到save_path中
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    i = 1
    for name in os.listdir(img_path):
        if i % 4 == 0:  # 将4的倍数的图片选中 即选中200张图片
            # print('name:\n', name)
            img = Image.open(os.path.join(img_path, name))
            img.save(os.path.join(save_path, name))
        i += 1


def removeSameImg(img_path1, img_path2, save_path):
    """
        功能：
            - 将path1中与path2中名字不同的图像写入save_path
        参数：
            img_path1: 待处理的图片目录
            img_path2: 进行比对的图片目录
            save_path: 指定的存储目录
        返回：
            path1处理后的图片输入到save_path中
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for name1 in os.listdir(img_path1):
        flag = True
        for name2 in os.listdir(img_path2):
            if name1 == name2:
                flag = False
                break
        if flag:
            img = Image.open(os.path.join(img_path1, name1))
            img.save(os.path.join(save_path, name1))


def binary2edge(mask_path, save_path):
    """
        功能：
            - 将mask图像的边缘提取出来并保存
        参数：
            mask_path：待转换的mask图像目录
            save_path：图像保存目录
        返回：
            - NULL
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for maskPath in os.listdir(mask_path):
        print(os.path.join(mask_path, maskPath))
        mask = cv2.imread(os.path.join(maskPath, mask_path), cv2.IMREAD_GRAYSCALE)
        print(mask)
        ret, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # if <0, pixel=0 else >0, pixel=255
        mask_edge = cv2.Canny(mask_binary, 10, 150)
        print(mask_edge)
        cv2.imwrite(os.path.join(save_path, maskPath), mask_edge)


def constructDataset(src_path, save_path, num):
    """
        功能：
            - 构建指定数目的数据集
        参数：
            src_path：源数据集
            save_path：图像保存目录
        返回：
            - NULL
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    i = 1
    j = 0
    for src in os.listdir(src_path):
        j += 1
        if j % 2 == 0:
            continue
        print(src)
        t = cv2.imread(os.path.join(src_path, src), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(save_path, src), t)
        i += 1
        if i == num:
            break


if __name__ == '__main__':
    # 构建数据集
    # src_path = r'E:\Pycharm\project\COVID19\dataset\findlDataset\tt'
    # save_path = r'E:\Pycharm\project\COVID19\dataset\findlDataset\test\label'
    # temp_path = r'E:\Pycharm\project\COVID19\dataset\findlDataset\val\label'
    # # constructDataset(src_path, save_path, 490/2)
    # if not os.path.exists(temp_path):
    #     os.mkdir(temp_path)
    # for src in os.listdir(src_path):
    #     flag = None
    #     for save in os.listdir(save_path):
    #         # print(save)
    #         if save == src:
    #             print(save)
    #             flag = save
    #             break
    #     if flag == None: # save中不存在该照片
    #         t = cv2.imread(os.path.join(src_path, src), cv2.IMREAD_GRAYSCALE)
    #         print(os.path.join(temp_path, src))
    #         cv2.imwrite(os.path.join(temp_path, src), t)

    # 通过lungmask图像,裁剪掉src图像和mask图像多余部分
    # src_path = r'E:\Pycharm\project\COVID19\dataset\finalData\img'
    # mask_path = r'E:\Pycharm\project\COVID19\dataset\finalData\mask'
    # lungmask_path = r'E:\Pycharm\project\COVID19\dataset\finalData\lung'
    # save_path = r'E:\Pycharm\project\COVID19\dataset\finalData\res'
    # cutImg(src_path, lungmask_path, mask_path, save_path)

    # 将mask图像处理成二值图像 src为mask文件所在位置 save_test为文件处理后像保存的位置
    # img_path = r'G:\scientific research\COVID19\new_COVID19_copy\res\label128'
    # save_path = r'G:\scientific research\COVID19\new_COVID19_copy\res\labelNew128'
    # binaryMask(img_path, save_path)

    # 将jpg转换成png
    # img_path = 'G:/scientific research/COVID19/NIT/val/img'
    # save_path = 'G:/scientific research/COVID19/NIT/valPNG'
    # convertPNG(img_path, save_path)

    # 将png转换成jpg
    # img_path = 'G:/scientific research/COVID19/NIT/test/label'
    # save_path = 'G:/scientific research/COVID19/NIT/test/labelJPG'
    # convertJPG(img_path, save_path)

    # 将png图像重新编号
    # srcImgPath = r'G:\scientific research\SCI_830\data\iChallenge-PM\test\imgs'
    # save_path = r'G:\scientific research\SCI_830\data\iChallenge-PM\test\tt'
    # renamePic(srcImgPath, save_path)

    # 将图像resize到128x128
    img_path = r'G:\scientific research\SCI_830\result\Covid19(all)\weighted_ours'
    save_path = r'G:\scientific research\SCI_830\result\Covid19(all)\weighted_ours1'
    resize(img_path, save_path)

    # 从train中选择200张图片作为val
    # img_path = 'G:/scientific research/COVID19/NIT/train/img128'
    # save_path = 'G:/scientific research/COVID19/NIT/val/img'
    # selectImg(img_path, save_path)

    # 从train中筛选出与val不同的图片
    # img_path1 = 'G:/scientific research/COVID19/NIT/train/img128'
    # img_path2= 'G:/scientific research/COVID19/NIT/val/img'
    # save_path = 'G:/scientific research/COVID19/NIT/trainR/img'
    # removeSameImg(img_path1, img_path2, save_path)

    # 将mask图像的边缘提取出来并保存
    # mask_path = r'E:\Pycharm\project\COVID19\dataset\augData\test\labelAug'
    # save_path = r'E:\Pycharm\project\COVID19\dataset\augData\test\edge'
    # binary2edge(mask_path, save_path)
    # import json
    # predN = 3
    # Feq = [3, 4, 5, 6]
    #
    # d = dict([('PredN', predN), ('Feq', Feq)])
    # d_json = json.dumps(d)
    # print(d_json)
