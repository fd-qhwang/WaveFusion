"""
这个脚本将 vaa3d 能够识别的 v3draw 格式的图像数据转换成一张张图片序列
"""

import struct
import os
import numpy as np
import cv2, ctypes
from tools.image_3D_io import load_image_3d
from constant import ImageSuffixes

def decode_image_v3draw(file_name, image_root_save = None, byteorder='little', height = None, width = None, offset = 27):
    """
    从名为 file_name 的 v3draw 文件中解码数据，分片读取，每片数据的高宽为 height、width，并将解码得到的图像数据保存在 image_root_save 中
    :param file_name: 二进制文件名
    :param image_root_save: 图像保存路径
    :param height: 高
    :param width: 宽
    # :param nbytes: 每个数据的字节长度
    :param byteorder: 'big' or 'little'
    :return:
    """
    assert os.path.isfile(file_name) and file_name.endswith('v3draw')
    if image_root_save == None:
        raise ValueError('没有制定解码后图片保存路径')
    if not os.path.isdir(image_root_save):
        os.makedirs(image_root_save)
    f = open(os.path.join(file_name), 'rb')
    records = f.read()
    f.close()

    if byteorder == 'big' or byteorder == '>' or byteorder == '!':
        byteorder = '>'
    elif byteorder == 'little' or byteorder == '<':
        byteorder = '<'

    width_, height_, depth, channel = struct.unpack_from(byteorder + 'IIII', records, offset)
    print(width_, height_, depth, channel)
    height = height_ if height == None else height
    width = width_ if width == None else width
    offset += 16

    current = 0
    nslice = 0
    row = 0
    col = 0
    slice = np.zeros((height, width))
    pixel = struct.unpack_from(byteorder + 'B', records, offset)[0]

    while True:
        offset += 1
        current += 1
        slice[height - row - 1,col] = pixel
        if col == (width - 1) and row == (height - 1):
            col = 0
            row = 0
            current = 0
            cv2.imwrite(os.path.join(image_root_save, str(nslice).zfill(6)) + '.tiff', slice)
            nslice += 1
            print('have got {} images, current = {}'.format(nslice, current))
            slice = np.zeros((height, width))
        elif col == (width - 1):
            col = 0
            row += 1
        else:
            col += 1
        try:
            pixel = struct.unpack_from(byteorder + 'B', records, offset)[0]
        except:
            print('have got {} images, current = {}'.format(nslice, current))
            break

def encode_to_v3draw(image_path, v3draw_file_name, byteorder='little'):
    """
    将二维图像序列编码保存为 Vaa3D 可以处理的 v3draw 文件
    :param image_root_save:  二维图像序列保存路径
    :param v3draw_file_name: 生成的 v3draw 文件名称
    :param byteorder: 编码顺序
    :return:
    """

    if byteorder == 'big' or byteorder == '>' or byteorder == '!':
        byteorder = '>'
    elif byteorder == 'little' or byteorder == '<':
        byteorder = '<'

    #head_info = b'data_produced_by_Qiufu_L' + b'L'
    head_info = b'raw_image_stack_by_hpeng' + b'L'
    image = load_image_3d(image_root = image_path)
    image = image[:,::-1,:]
    f = open(v3draw_file_name, 'wb')
    depth, height, width = image.shape
    prebuffer = ctypes.create_string_buffer(43)
    struct.pack_into(byteorder + '25s2B4I', prebuffer, 0, *(head_info, 1, 0, width, height, depth, 1))
    f.write(prebuffer)

    data_size = depth * height * width
    print('encoding {} ==> {} MB'.format(image_path, data_size / 1024 / 1024))
    prebuffer = ctypes.create_string_buffer(data_size)
    struct.pack_into(byteorder + '{}B'.format(data_size), prebuffer, 0, *image.flatten())
    f.write(prebuffer)

    f.close()


def encode_to_v3draw_slice(image_path, v3draw_file_name, byteorder='little', flip = '10'):
    """
    将二维图像序列编码保存为 Vaa3D 可以处理的 v3draw 文件
    :param image_root_save:  二维图像序列保存路径
    :param v3draw_file_name: 生成的 v3draw 文件名称
    :param byteorder: 编码顺序
    :param flip: 是否翻转图像; 00: 不翻转，
                              01: 翻转第二个轴，
                              10: 翻转第一个轴，
                              11: 两个轴都翻转
    :return:
    """

    if byteorder == 'big' or byteorder == '>' or byteorder == '!':
        byteorder = '>'
    elif byteorder == 'little' or byteorder == '<':
        byteorder = '<'

    image_name_list = os.listdir(image_path)
    image_name_list.sort()
    image_name_list_ = []
    for image_name in image_name_list:
        _, suffix = os.path.splitext(image_name)
        if suffix not in ImageSuffixes:
            continue
        image_name_list_.append(image_name)

    f = open(v3draw_file_name, 'wb')
    head_info_flag = False
    depth = len(image_name_list_)
    for index,image_name in enumerate(image_name_list_):
        image = cv2.imread(os.path.join(image_path, image_name), 0)
        if flip == '00': pass
        elif flip == '10': image = image[::-1,:]
        elif flip == '01': image = image[:,::-1]
        elif flip == '11': image = image[::-1, ::-1]
        else: raise NotImplementedError('"flip" should in {"00","01","10","11"}')
        if len(image.shape) == 2:
            height, width = image.shape
        elif len(image.shape) == 3:
            _, height, width = image.shape
        else:
            raise NotImplementedError('这个程序处理不了其他类型的图像，图像数据应只有空间维度和通道维度')
        if not head_info_flag:
            #head_info = b'data_produced_by_Qiufu_L' + b'L'
            head_info = b'raw_image_stack_by_hpeng' + b'L'
            prebuffer = ctypes.create_string_buffer(43)
            struct.pack_into(byteorder + '25s2B4I', prebuffer, 0, *(head_info, 1, 0, width, height, depth, 1))
            f.write(prebuffer)
            head_info_flag = True


        data_size = height * width
        if index % 10 == 0:
            print('encoding {}-{} ({}/{}) ==> {} MB'.format(image_path, image_name, index, depth, data_size / 1024 / 1024))
        prebuffer = ctypes.create_string_buffer(data_size)
        struct.pack_into(byteorder + '{}B'.format(data_size), prebuffer, 0, *image.flatten())
        f.write(prebuffer)
    f.close()


if __name__ == '__main__':
    image_path = '/media/liqiufu/NeuronData_4TB/Mouse_15107/Neuron_image_3d/N041'
    v3draw_file_name = '/media/liqiufu/NeuronData_4TB/Mouse_15107/Neuron_image_3d/N041.v3draw'
    encode_to_v3draw_slice(image_path, v3draw_file_name, byteorder = 'little')