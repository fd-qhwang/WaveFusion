"""
这个脚本负责3D图像的导入和导出
"""
import os, cv2, sys
import numpy as np
from constant import Image2DName_Length
from tools.printer import print_my
from datetime import datetime

IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png', '.tif']

def load_image_3d(image_root):
    """
    从文件夹 image_root 中加载其中包含的 3D 图像数据
    :param image_root: 一个保存二维图像序列的路径，这些二维图像是一个三维图像的分片表示
    :return:
    """
    image_name_list = os.listdir(image_root)
    image_name_list.sort()
    image_3d = []
    for image_name in image_name_list:
        _, suffix = os.path.splitext(image_name)
        if suffix not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(os.path.join(image_root, image_name), 0)
        image_3d.append(image)
    return np.array(image_3d)

def save_image_3d(image_3d, image_save_root, dim = 0, suffix = '.tiff'):
    """
    这个函数将三维矩阵保存为对应张数的二维图像
    :param image_3d: np.multiarray, 三维矩阵
    :param image_save_root: string, 保存路径
    :param dim: int, 三维矩阵的切片维度
    :param suffix: string, 保存的二维图像的后缀名
    :return:
    """
    assert dim < len(image_3d.shape)
    assert suffix in IMAGE_SUFFIXES
    if not os.path.isdir(image_save_root):
        os.makedirs(image_save_root)
    #shape_ = [image_3d.shape[i] for i in range(len(image_3d.shape)) if image_3d.shape[i] != 1]
    #image_3d = image_3d.reshape(shape_)
    #shape_ = [image_3d.shape[i] for i in range(len(image_3d.shape)) if i != dim]
    #shape_.insert(0, image_3d.shape[dim])
    #print(image_3d.shape)
    if len(os.listdir(image_save_root)):
        os.system('rm ' + os.path.join(image_save_root,'*'))
    depth = image_3d.shape[dim]
    for index in range(depth):
        image_full_name = os.path.join(image_save_root, str(index).zfill(Image2DName_Length)) + suffix
        flag = cv2.imwrite(image_full_name, image_3d[index])
        if flag == False:
            print('{} to save {}-th image in {}'.format(flag, index, image_save_root))
            break
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\r{}-------- saving {} / {} 2D image ...'.format(time, index, depth))
        sys.stdout.flush()
    time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
    sys.stdout.write('\n{}-------- finish saving !\n'.format(time))

if __name__ == '__main__':
    image_root = '/home/li-qiufu/PycharmProjects/MyDataBase/test_result_32/DataBase_16/angle_0/000089/label_pre'
    image_root_s = '/home/li-qiufu/PycharmProjects/MyDataBase/test_result_32/DataBase_16/angle_0/000089/label_pre_1'
    image_3d = load_image_3d(image_root = image_root)
    image_3d[image_3d != 1] = 0
    save_image_3d(image_3d = image_3d, image_save_root = image_root_s)