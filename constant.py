"""
这个脚本保存项目涉及到的一些常数，涉及到路径配置的常数在使用时需要相应修改
"""
import torch, os

#神经元图像像素类别数，只有前景类和背景类两种
NUM_CLASSES = 2

# 神经元图像数据是以二维图像序列形式保存的，其后缀名是一致的
ImageSuffixes = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

"""
# 处理超大尺寸(如完整鼠脑图像)的神经元图像时，囿于电脑硬件配置可能会很慢，可以将图像大块切分后再处理
# 需要被划分的尺寸，当神经元图像的任意一个维度大小超过这个尺寸的对应维度大小时候，将其进行划分
ImageSize_NoMoreThan = (256, 1024, 1024)    # 当图像任何一个尺寸超过这个大小时候，则认为其是超大型的，不适合低配置的计算机处理；
                                            # 将该图像划分后处理，划分大小为 ImageSice_PartitionTo
ImageSize_PartitionTo = (192, 768, 768)     # 将过大的图像划分的目标大小
#ImageSize_NoMoreThan = (128, 512, 512)
#ImageSize_PartitionTo = (96, 384, 384)
ImageSize_Overlap = (0, 0, 0)             # 超大尺寸图像在划分时候相邻子图像之间的重叠大小
"""

Image2DName_Length = 6
Image3DName_Length = 4

#输入网络进行训练或去噪处理的图像块大小
Block_Size = (32, 128, 128)     #(z, y, x) -- depth, height, width
Block_Size_ = 32 * 128 * 128

WAVENAME_LIST = ['none', 'haar', 'bior2.2', 'bior3.3', 'bior4.4', 'bior5.5', 'db2', 'db3', 'db4', 'db5', 'db6']

DataTrain_Root = '/data/liqiufu/DATA/Neuron_DataBase_for_Denoiser/DataBase_5_new'
#训练数据保存路径，其中保存的是 neuronal cubes
TrainSource = os.path.join(DataTrain_Root, 'train.txt') #训练数据
TestSource = os.path.join(DataTrain_Root, 'test.txt')   #测试数据
Mean_TrainData = 0.029720289547802054   # 这是基于 BigNeuron 数据生成的图像块的均值和方差
Std_TrainData = 0.04219472495471814

#完整神经元图像保存路径
DataNeuron_Root = '/data/liqiufu/DATA/Neuron_DataBase_for_Denoiser/DataBase_1_resized' if torch.cuda.is_available() \
    else '/media/liqiufu/NeuronData_4TB/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1_resized'
#完整神经元图像保存路径，其中保存的是 neuronal images

RESULT_SAVE_PATH = '/data/liqiufu/DATA/Neuron_DataBase_for_Denoiser/Neuron_Denoised' if torch.cuda.is_available() \
    else '/media/liqiufu/NeuronData_4TB/MyDataBase/Neuron_DataBase_for_Denoiser/Neuron_Denoised'

NeuronNet = {'neuron_segnet', 'neuron_unet_v1', 'neuron_unet_v2',
             'neuron_wavesnet_v1', 'neuron_wavesnet_v2', 'neuron_wavesnet_v3', 'neuron_wavesnet_v4'}

ROOT_STANDARD = '/data/liqiufu/DATA/Neuron_DataBase_for_Denoiser/DataBase_1_resized' if torch.cuda.is_available() \
    else '/media/liqiufu/NeuronData_4TB/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_2'

MODEL_ROOT = '/data/liqiufu/Projects/Neuron_Denoiser/weight' if torch.cuda.is_available() else NotImplementedError
#预训练好的深度网络模型保存路径；请在GPU中使用，不建议在本地电脑操作

NEURON_NAME_LIST = ['000000', '000011', '000025', '000036', '000047', '000059', '000072', '000083', '000094',
                    '000001', '000012', '000026', '000037', '000048', '000060', '000073', '000084', '000095',
                    '000002', '000013', '000027', '000038', '000049', '000061', '000074', '000085',
                    '000003', '000015', '000028', '000039', '000050', '000064', '000075', '000086',
                    '000004', '000016', '000029', '000040', '000051', '000065', '000076', '000087',
                    '000005', '000017', '000030', '000041', '000052', '000066', '000077', '000088',
                    '000006', '000018', '000031', '000042', '000053', '000067', '000078', '000089',
                    '000007', '000019', '000032', '000043', '000054', '000068', '000079', '000090',
                    '000008', '000022', '000033', '000044', '000055', '000069', '000080', '000091',
                    '000009', '000023', '000034', '000045', '000056', '000070', '000081', '000092',
                    '000010', '000024', '000035', '000046', '000057', '000071', '000082', '000093']

NEURON_NAME_TEST = ['000002', '000006', '000013', '000020', '000025', '000029',
                    '000033', '000036', '000040', '000043', '000049', '000059',
                    '000064', '000068', '000072', '000075', '000078', '000082',
                    '000086', '000089', '000093', '000008', '000009', '000011',
                    '000012', '000015', '000018', '000031', '000046', '000047',
                    '000050', '000058', '000063']


NEURON_NAME_NOT_COMPARE = ['000006', '000008', '000013', '000058', '000063', '000072', '000089', '000093', '000068', '000059', '000064']

NEURON_NOT_IN_TRAIN_TEST = ['000008', '000009', '000011', '000012', '000015', '000018',
                            '000031', '000046', '000047', '000050', '000058', '000063']
#以上神经元名称是将BigNeuron中神经元图像按数字顺序重排后的名称

FUNC = {
    'APP2':         '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_tracing/Vaa3D_Neuron2/libvn2.so -f app2',
    'NeuroGPSTree': '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_tracing/HUST_NeuroGPSTree/libNeuroGPSTree.so -f tracing_func',
    'MST_Tracing':  '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_tracing/MST_tracing/libneurontracing_mst.so -f trace_mst',
    'XY_3D_TreMap': '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_tracing/TReMap/libneurontracing_mip.so -f trace_mip',
    'snake':        '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_tracing/Vaa3D-FarSight_snake_tracing/libsnake_tracing.so -f snake_trace',
}
# FUNC 中的命令参数需要根据个人电脑中 Vaa3D 的配置路径进行修改
# 由于我的办公电脑中只有以上几个算法能使用批量化操作，且这几个追踪算法运行速度较快，因此我们只使用以上几个追踪方法进行对比