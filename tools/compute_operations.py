import torch

class OperationNumber():
    """
    统计网络处理一个数据所需的加乘操作数量
    """
    def __init__(self, net):
        """
        :param net: 已经加载预训练参数的网络模型
        这个待处理的网络输入的每一个 batch 包含不同的两部分图像，前一半是含噪图像数据 x + e，后一半是原始干净图像数据 x
        """
        self.net = net
        self.feature_name_list = list()
        self.layer_name_list = list()

        self.op_number_i_list = list()   # 保存当前层所需的操作数量

    def _is_leaf(self, module):
        """
        判断网络的一个模块是否是叶子节点，即层 layer
        :param module:
        :return:
        """
        if sum(1 for x in module.children()) == 0:
            return True
        elif module._get_name().startswith('BlurPool'):
            return True
        else:
            sum_padding = 0
            for name, layer in module.named_children():
                if 'padding' in name:
                    sum_padding += 1
            if sum_padding == sum(1 for x in module.children()):
                return True
        return False

    def hook_all_layers(self):
        """
        :return:
        """
        hook_OrderedDict = list()

        def hook(module, inputdata, output):
            """
            钩子函数，计算统计量
            :param module: 模块，层
            :param inputdata: 输入数据，元组，长度可能不是 1，保存张量
            :param output: 输出数据，元组
            :return:
            """
            m_name = module._get_name()
            if m_name.startswith('Conv'):   # 层类型，若是卷积保存其步长数
                self.layer_name_list.append(m_name+'_i{}o{}k{}s{}'.format(module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0]))
            elif m_name.startswith('MaxPool') or m_name.startswith('AvgPool'):
                self.layer_name_list.append(m_name + '_k{}s{}'.format(module.kernel_size, module.stride))
            else:
                self.layer_name_list.append(module._get_name())
            #如果当前模块是叶子节点，则计算其所需的操作数量
            op_no = get_operation_NO(module, inputdata, output)
            self.op_number_i_list.append(op_no)

        def the_leaf(model, pre_name=''):
            """
            通过递归形式保存网络中所有层 layer 的名称，并给每一层后添加钩子函数
            :param model:
            :param pre_name:
            :return:
            """
            for index, param in enumerate(model.named_children()):                          # 逐个处理网络中的每一个模块
                name, layer = param
                name = name if pre_name == '' else pre_name + '.' + name
                if self._is_leaf(layer):                                                    # 若是层 layer
                    if layer._get_name().startswith('Dropout') or layer._get_name().startswith('Lambda'):
                        continue
                    hook_OrderedDict.append(layer.register_forward_hook(hook))              # 添加钩子函数
                    self.feature_name_list.append(name)                                     # 保存层名称
                else:
                    the_leaf(layer, pre_name = name)
        the_leaf(self.net, pre_name = '')

    def run(self, shape = (1,3,224,224)):
        input = torch.randn(shape)
        self.hook_all_layers()
        self.net(input)
        assert len(self.layer_name_list) == len(self.op_number_i_list), \
            'len(self.layer_name_list) = {}, len(self.op_number_i_list) = {}'\
                .format(len(self.layer_name_list),len(self.op_number_i_list))
        sum_op = 0
        for index, name in enumerate(self.layer_name_list):
            #feature_name = self.feature_name_list[index]
            text = '{} ==> {}'.format(name.ljust(48), self.op_number_i_list[index])
            sum_op += self.op_number_i_list[index]
            print(text)
        print('-'*100)
        print('-'*100)
        print('网络所需加乘操作数量总数为：{} ==> {} K ==> {} M ==> {} G'.
              format(sum_op, sum_op / (10**3), sum_op / (10**6), sum_op / (10**9)))

    def counter(self):
        params = list(self.net.parameters())
        k = 0
        for i in params:
            l = 1
            #print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
            #print("该层参数和：" + str(l))
            k = k + l
        print("总参数数量和：{} ==> {} K ==> {} M".format(k, k / 1000, k / 1000 / 1000))


def get_operation_NO(layer, inputdata, output):
    """
    计算 module 处理 inputdata 所需的加乘操作数量
    :param layer: module
    :param inputdata: input data
    :return:
    """
    type_name = layer._get_name()
    #print(type_name)
    if type_name in ['Conv2d']:
        x = inputdata[0]
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w // layer.groups

    elif type_name in ['DWT_3D_tiny']:
        x = inputdata[0]
        channels = x.size()[1]
        out_d = int((x.size()[2] + layer.kernel_size // 2) // layer.stride + 1)
        out_h = int((x.size()[3] + layer.kernel_size // 2) // layer.stride + 1)
        out_w = int((x.size()[4] + layer.kernel_size // 2) // layer.stride + 1)
        delta_ops = layer.kernel_size ** 3 * channels * out_d * out_h * out_w

    elif type_name in ['DWT_3D']:
        x = inputdata[0]
        channels = x.size()[1]
        out_d = int((x.size()[2] + layer.kernel_size // 2) // layer.stride + 1)
        out_h = int((x.size()[3] + layer.kernel_size // 2) // layer.stride + 1)
        out_w = int((x.size()[4] + layer.kernel_size // 2) // layer.stride + 1)
        delta_ops = 8 * layer.kernel_size ** 3 * channels * out_d * out_h * out_w

    elif type_name in ['IDWT_3D']:
        x = inputdata[0]
        channels = x.size()[1]
        out_d = int((x.size()[2] + layer.kernel_size // 2) + 1)
        out_h = int((x.size()[3] + layer.kernel_size // 2) + 1)
        out_w = int((x.size()[4] + layer.kernel_size // 2) + 1)
        delta_ops = 8 * layer.kernel_size ** 3 * channels * out_d * out_h * out_w

    elif type_name in ['Hardshrink']:
        x = inputdata[0]
        delta_ops = x.numel()

    elif type_name in ['Conv3d',]:
        x = inputdata[0]
        out_d = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1)
        out_h = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1)
        out_w = int((x.size()[4] + 2 * layer.padding[2] - layer.kernel_size[2]) // layer.stride[2] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * layer.kernel_size[2] * out_d * out_h * out_w // layer.groups

    elif type_name in ['Conv3d']:
        #print(type(inputdata), len(inputdata))
        #print(type(output), len(output))
        x = inputdata[0]
        #print(x.size())
        out_d = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1)
        out_h = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1)
        out_w = int((x.size()[4] + 2 * layer.padding[2] - layer.kernel_size[2]) // layer.stride[2] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * layer.kernel_size[2] * out_d * out_h * out_w // layer.groups

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        x = inputdata[0]
        delta_ops = x.numel()

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        x = inputdata[0]
        in_w = x.size()[2]
        in_h = x.size()[3]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
        out_h = int((in_h + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
        delta_ops = x.size()[1] * out_w * out_h * kernel_ops
    ### ops_pooling
    elif type_name in ['AvgPool3d', 'MaxPool3d', ]:
        x = inputdata[0]
        in_d = x.size()[2]
        in_w = x.size()[3]
        in_h = x.size()[4]
        kernel_ops = layer.kernel_size * layer.kernel_size * layer.kernel_size
        out_d = int((in_d + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
        out_h = int((in_h + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
        delta_ops = x.size()[1] * out_d * out_w * out_h * kernel_ops

    elif type_name in ['AdaptiveAvgPool2d', 'AdaptiveAvgPool3d']:
        x = inputdata[0]
        delta_ops = x.numel()

    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel()
        bias_ops = layer.bias.numel()
        delta_ops = weight_ops + bias_ops

    elif type_name in ['BatchNorm2d', 'BatchNorm3d']:
        x = inputdata[0]
        normalize_ops = x.numel()
        scale_shift = normalize_ops
        delta_ops = normalize_ops + scale_shift

    ### ops_nothing
    elif type_name in ['Dropout2d', 'DropChannel', 'Dropout']:
        delta_ops = 0

    elif type_name in ['SpatialGate_CT_C']:
        x = inputdata[0]
        delta_ops = 2 * x.numel()

    elif type_name in ['DWT2_2D']:
        x = inputdata[0]
        channels = x.size()[1]
        out_h = int((x.size()[2] + 2 * layer.pad_sizes[0] - layer.kernel_size) // layer.stride + 1)
        out_w = int((x.size()[3] + 2 * layer.pad_sizes[1] - layer.kernel_size) // layer.stride + 1)
        delta_ops = channels * layer.kernel_size * layer.kernel_size * out_h * out_w // layer.groups

    ### unknown layer type
    else:
        delta_ops = 0
        print('unknown layer type: %s' % type_name)

    return delta_ops



if __name__ == '__main__':
    from networks.neuron_net import Neuron_WaveSNet_V1, Neuron_WaveSNet_V2, Neuron_WaveSNet_V3, Neuron_WaveSNet_V4
    from networks.neuron_net import Neuron_SegNet, Neuron_UNet_V1, Neuron_UNet_V2
    net = Neuron_UNet_V1()
    net = Neuron_SegNet()
    #net = Neuron_WaveSNet_V4(wavename = 'db2')
    op_no = OperationNumber(net)
    op_no.run(shape = (1,1,32,128,128))
    op_no.counter()