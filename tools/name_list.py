import os

root = '/data/liqiufu/DATA/Neuron_DataBase_for_Denoiser/DataBase_5_new'
train_txt = os.path.join(root, 'train.txt')
test_txt = os.path.join(root, 'test.txt')

def name_list():
    train_info = open(train_txt, 'w')
    test_info = open(test_txt, 'w')
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    for index, neuron_name in enumerate(neuron_name_list):
        neuron_path = os.path.join(root, neuron_name)
        if not os.path.isdir(neuron_path):
            continue
        block_name_list = os.listdir(neuron_path)
        block_name_list.sort()
        for block_name in block_name_list:
            block = os.path.join(neuron_path, block_name)
            if not os.path.isdir(block):
                continue
            if (index + 1) % 4 == 0:
                test_info.write(os.path.join(neuron_name, block_name) + '\n')
            else:
                train_info.write(os.path.join(neuron_name, block_name) + '\n')
    train_info.close()
    test_info.close()

if __name__ == '__main__':
    name_list()