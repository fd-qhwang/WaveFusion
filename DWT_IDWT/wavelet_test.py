from DWT_IDWT.DWT_IDWT_layer import DWT_2D as DWT_matrix
from DWT_IDWT.DWT_IDWT_CS import DWT_2D as DWT_cov

import cv2, torch
import datetime

if __name__ == '__main__':
    image_full_name = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    print(image.shape)
    image = image[0:512, 0:512, :]
    print(image.shape)
    height, width, channel = image.shape
    image_tensor = torch.tensor(image)
    image_tensor = image_tensor.transpose(dim0 = 0, dim1 = 2)
    image_tensor = image_tensor.unsqueeze(dim = 0)
    image_tensor = image_tensor.float()
    image_tensor = image_tensor.cuda() if torch.cuda.is_available() else image_tensor

    N = 1000
    t0 = datetime.datetime.now()
    down_sampling = DWT_matrix(wavename = 'haar')
    down_sampling = down_sampling.cuda() if torch.cuda.is_available() else down_sampling
    for index in range(N):
        LL, LH, HL, HH = down_sampling(image_tensor)
    t1 = datetime.datetime.now()
    print('{} times of DWT_matrix took {} secs.'.format(N, t1 - t0))

    t0 = datetime.datetime.now()
    down_sampling = DWT_cov(wavename = 'haar', in_channels = 3)
    down_sampling = down_sampling.cuda() if torch.cuda.is_available() else down_sampling
    for index in range(N):
        LL, LH, HL, HH = down_sampling(image_tensor)
    t1 = datetime.datetime.now()
    print('{} times of DWT_conv took {} secs.'.format(N, t1 - t0))