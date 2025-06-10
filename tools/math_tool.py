"""
这个脚本处理一些数学几何问题
"""
import math

def line_parameters_2d(point0, point1):
    """
    获取二维平面中过两点直线的一般方程的参数 A, B, C，这里 Ax + By + C = 0
    :param point0:
    :param point1:
    :return:
    """
    assert len(point0) == 2 and len(point1) == 2
    x0, y0 = point0
    x1, y1 = point1
    assert x0 != x1 and y0 != y1
    A = (y1 - y0) * 1.0
    B = (x0 - x1) * 1.0
    C = (x1 * y0 - x0 * y1) * 1.0
    return A, B, C

def line_parameters_3d(point0, point1):
    """
    获取三维空间中过两点直线的一般方程的参数 X, Y, Z, x0, y0, z0，这里参数 x = Xt+x0, y = Yt+y0, z = Zt + z0
    :param point0:
    :param point1:
    :return:
    """
    assert len(point0) == 3 and len(point1) == 3
    x0, y0, z0 = point0
    x1, y1, z1 = point1
    assert not (x0 == x1 and y0 == y1 and z0 == z1)
    X = (x1 - x0) * 1.0
    Y = (y1 - y0) * 1.0
    Z = (z1 - z0) * 1.0
    return X, Y, Z, x0, y0, z0

def get_connect_coordinate_2d(point0, point1):
    """
    获取二维平面中两个点连线附近的坐标点
    :param point0:
    :param point1:
    :return:
    """
    assert len(point0) == 2 and len(point1) == 2
    x0, y0 = point0
    x1, y1 = point1
    if x0 == x1 and y0 == y1:
        return []
    if x0 == x1:
        step = 1 if y0 < y1 else -1
        return [(x0, y) for y in range(y0+1, y1, step)]
    if y0 == y1:
        step = 1 if x0 < x1 else -1
        return [(x, y0) for x in range(x0+1, x1, step)]
    A,B,C = line_parameters_2d(point0, point1)
    #print(A,B,C)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    xx, yy = x0, y0
    the_coordinate = []
    while True:
        temp_list = [(xx, yy+step_y), (xx+step_x, yy), (xx+step_x, yy+step_y)]
        d_list = [abs(A*x+B*y+C) / math.sqrt(A*A+B*B) for (x,y) in temp_list]
        xx,yy = temp_list[d_list.index(min(d_list))]
        if xx == x1 and yy == y1:
            break
        #the_coordinate.append((xx,yy))
        the_coordinate.extend(temp_list)
    return the_coordinate

def get_connect_coordinate_3d(point0, point1):
    """
    获取三维平面中两个点连线附近的坐标点
    :param point0:
    :param point1:
    :return:
    """
    assert len(point0) == 3 and len(point1) == 3
    x0, y0, z0 = point0
    x1, y1, z1 = point1
    if x0 == x1 and y0 == y1 and z0 == z1:
        return []
    if x0 == x1:
        the_coordinate = get_connect_coordinate_2d((y0, z0), (y1, z1))
        return [(x0, yy, zz) for (yy,zz) in the_coordinate]
    if y0 == y1:
        the_coordinate = get_connect_coordinate_2d((x0, z0), (x1, z1))
        return [(xx, y0, zz) for (xx,zz) in the_coordinate]
    if z0 == z1:
        the_coordinate = get_connect_coordinate_2d((x0, y0), (x1, y1))
        return [(xx, yy, z0) for (xx,yy) in the_coordinate]
    X, Y, Z, x_0, y_0, z_0 = line_parameters_3d(point0, point1)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    step_z = 1 if z0 < z1 else -1
    xx, yy, zz = x0, y0, z0
    the_coordinate = []
    while True:
        temp_list = [(xx, yy, zz + step_z), (xx, yy + step_y, zz), (xx, yy + step_y, zz + step_z),
                     (xx + step_x, yy, zz), (xx + step_x, yy, zz + step_z), (xx + step_x, yy + step_y, zz),
                     (xx + step_x, yy + step_y, zz + step_z)]
        T_list = [(x * X - X * x_0 + y * Y - Y * y_0 + z * Z - Z * z_0) / (X*X + Y*Y + Z*Z)
                  for (x,y,z) in temp_list]
        M_list = [(X*t+x_0, Y*t+y_0, Z*t+z_0) for t in T_list]
        D_list = [math.sqrt((temp_list[i][0] - M_list[i][0]) ** 2 + (temp_list[i][1] - M_list[i][1]) ** 2 + (temp_list[i][2] - M_list[i][2]) ** 2)
                  for i in range(len(temp_list))]
        xx, yy, zz = temp_list[D_list.index(min(D_list))]
        if xx == x1 and yy == y1 and zz == z1:
            break
        #the_coordinate.append((xx, yy, zz))
        the_coordinate.extend(temp_list)
    return the_coordinate

def get_connect_coordinate_3d_new(point0, point1, radius = 2, resolution = [1,1,1/0.32]):
    """
    获取三维平面中两个点连线附近的坐标点
    :param point0:
    :param point1:
    :return:
    """
    assert len(point0) == 3 and len(point1) == 3
    radius = round(radius)
    assert radius > 0
    x0, y0, z0 = point0
    x1, y1, z1 = point1
    r_x = radius / resolution[0]
    r_y = radius / resolution[1]
    r_z = radius / resolution[2]
    if x0 == x1 and y0 == y1 and z0 == z1:
        return []
    x_min = min(x0, x1)
    x_max = max(x0, x1)
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    z_min = min(z0, z1)
    z_max = max(z0, z1)
    if x_max - x_min + 1 >= 2 * radius:
        x_a, x_b = x_min, x_max
    else:
        x_a = round(x_min / 2 - radius + x_max / 2)
        x_b = round(x_max / 2 + radius + x_min / 2)
    if y_max - y_min + 1 >= 2 * radius:
        y_a, y_b = y_min, y_max
    else:
        y_a = round(y_min / 2 - radius + y_max / 2)
        y_b = round(y_max / 2 + radius + y_min / 2)
    if z_max - z_min + 1 >= 2 * radius:
        z_a, z_b = z_min, z_max
    else:
        z_a = round(z_min / 2 - radius + z_max / 2)
        z_b = round(z_max / 2 + radius + z_min / 2)
    X, Y, Z, x_0, y_0, z_0 = line_parameters_3d(point0, point1)
    temp_list = [(x,y,z) for x in range(x_a+1, x_b)
                         for y in range(y_a+1, y_b)
                         for z in range(z_a+1, z_b)]
    T_list = [(x * X - X * x_0 + y * Y - Y * y_0 + z * Z - Z * z_0) / (X*X + Y*Y + Z*Z)
              for (x,y,z) in temp_list]
    M_list = [(X*t+x_0, Y*t+y_0, Z*t+z_0) for t in T_list]
    D_list = [math.sqrt((temp_list[i][0] - M_list[i][0]) ** 2 / (r_x ** 2) + (temp_list[i][1] - M_list[i][1]) ** 2  / (r_y ** 2)+ (temp_list[i][2] - M_list[i][2]) ** 2) / (r_z ** 2)
              for i in range(len(temp_list))]
    the_coordinate = [(x, y ,z) for ((x,y,z), r) in zip(temp_list, D_list) if r <= 1]
    return the_coordinate

def rotate_3d(point, angle_Z = 0, angle_Y = 0, angle_X = 0):
    """
    对空间中的三维坐标点 point 进行旋转
    :param point:
    :param angle_Z: 点 point 在XOY平面的投影与 X 轴夹角增加 angle_Z，即绕 Z 轴旋转 angle_theta 角度
    :param angle_Y: 点 point 在XOZ平面的投影与 X 轴的夹角增加 angle_Y，即绕 Y 轴旋转 angle_theta 角度
    :param angle_X: 点 point 在YOZ平面的投影与 Y 轴的夹角增加 angle_X，即绕 X 轴旋转 angle_theta 角度
    :return: 返回的点不一定是整数值，如果是进行整数坐标变换，需进行取整；若追求精度，可进行插值
    """
    assert len(point) == 3
    x0, y0, z0 = point
    if angle_Z == 0:
        x1, y1, z1 = x0, y0, z0
    else:
        x1 = x0 * math.cos(angle_Z) - y0 * math.sin(angle_Z)
        y1 = x0 * math.sin(angle_Z) + y0 * math.cos(angle_Z)
        z1 = z0

    if angle_Y == 0:
        x2, y2, z2 = x1, y1, z1
    else:
        x2 = x1 * math.cos(angle_Y) + z1 * math.sin(angle_Y)
        y2 = y1
        z2 = - x1 * math.sin(angle_Y) + z1 * math.cos(angle_Y)
    if angle_X == 0:
        x,y,z = x2,y2,z2
    else:
        x = x2
        y = y2 * math.cos(angle_X) - z2 * math.sin(angle_X)
        z = y2 * math.sin(angle_X) + z2 * math.cos(angle_X)
    return x,y,z

def distance(point0, point1):
    """
    计算两个点之间的距离
    :param point0:
    :param point1:
    :return:
    """
    assert len(point0) == len(point1)
    d = sum([(x-y)**2 for (x,y) in zip(point0, point1)])
    return math.sqrt(d)

def getshape_rotate_3d(shape, angle_Z = 0, angle_Y = 0, angle_X = 0):
    """
    对形如 shape 的所有空间坐标点以 point 点为原点进行旋转后，返回其最大外接立方体的形状参数
    :param shape: 原始形状
    :param angle_Z: 绕 Z 轴旋转的角度
    :param angle_Y: 绕 Y 轴旋转的角度
    :param angle_X: 绕 X 轴旋转的角度
    :return:
    """
    assert len(shape) == 3
    x07, y07, z07 = shape
    x00, y00, z00 = 0,   0,   0
    x01, y01, z01 = 0,   0,   z07
    x02, y02, z02 = 0,   y07, 0
    x03, y03, z03 = 0,   y07, z07
    x04, y04, z04 = x07, 0,   0
    x05, y05, z05 = x07, 0,   z07
    x06, y06, z06 = x07, y07, 0
    for i in range(8):
        x = locals()['x0' + str(i)]
        y = locals()['y0' + str(i)]
        z = locals()['z0' + str(i)]
        x,y,z = rotate_3d((x,y,z),angle_Z,angle_Y,angle_X)
        locals().update({'x1'+str(i) : x})
        locals().update({'y1'+str(i) : y})
        locals().update({'z1'+str(i) : z})
    #print(locals()['x1'+str(0)])
    x0 = min([locals()['x10'], locals()['x11'], locals()['x12'], locals()['x13'], locals()['x14'], locals()['x15'], locals()['x16'], locals()['x17']])
    x1 = max([locals()['x10'], locals()['x11'], locals()['x12'], locals()['x13'], locals()['x14'], locals()['x15'], locals()['x16'], locals()['x17']])
    y0 = min([locals()['y10'], locals()['y11'], locals()['y12'], locals()['y13'], locals()['y14'], locals()['y15'], locals()['y16'], locals()['y17']])
    y1 = max([locals()['y10'], locals()['y11'], locals()['y12'], locals()['y13'], locals()['y14'], locals()['y15'], locals()['y16'], locals()['y17']])
    z0 = min([locals()['z10'], locals()['z11'], locals()['z12'], locals()['z13'], locals()['z14'], locals()['z15'], locals()['z16'], locals()['z17']])
    z1 = max([locals()['z10'], locals()['z11'], locals()['z12'], locals()['z13'], locals()['z14'], locals()['z15'], locals()['z16'], locals()['z17']])
    #print(shape, (x1-x0), (y1-y0), (z1-z0))
    return (x1-x0), (y1-y0), (z1-z0)

def rotate_3d_inverse(point, angle_Z = 0, angle_Y = 0, angle_X = 0):
    """
    这个函数相当于函数 rotate_3d 的逆操作，它将点先绕 X 轴旋转 -angle_X 角度，再绕 Y 轴旋转 -angle_Y 角度，最后绕 Z 轴旋转 -angle_Z 角度
    :param point:
    :param angle_Z:
    :param angle_Y:
    :param angle_X:
    :return: 返回的点不一定是整数值，如果是进行整数坐标变换，需进行取整；若追求精度，可进行插值
    """
    assert len(point) == 3
    angle_Z = -angle_Z
    angle_Y = -angle_Y
    angle_X = -angle_X
    x0, y0, z0 = point
    if angle_X == 0:
        x1,y1,z1 = x0,y0,z0
    else:
        x1 = x0
        y1 = y0 * math.cos(angle_X) - z0 * math.sin(angle_X)
        z1 = y0 * math.sin(angle_X) + z0 * math.cos(angle_X)
    if angle_Y == 0:
        x2, y2, z2 = x1, y1, z1
    else:
        x2 = x1 * math.cos(angle_Y) + z1 * math.sin(angle_Y)
        y2 = y1
        z2 = - x1 * math.sin(angle_Y) + z1 * math.cos(angle_Y)
    if angle_Z == 0:
        x, y, z = x2, y2, z2
    else:
        x = x2 * math.cos(angle_Z) - y2 * math.sin(angle_Z)
        y = x2 * math.sin(angle_Z) + y2 * math.cos(angle_Z)
        z = z2
    return x,y,z

def get_coordinate_along_vector(point_start, vector, distance):
    """
    这个代码返回从初始坐标开始、沿着方向向量、经过确定长度后的坐标值
    :param point_start: 初始坐标
    :param vector: 方向向量
    :param distance: 确定的长度
    :return:
    """
    z_start, y_start, x_start = point_start
    z_vector, y_vector, x_vector = vector
    k = math.sqrt(distance ** 2 / (z_vector ** 2 + y_vector ** 2 + x_vector ** 2))
    return z_start + k * z_vector, y_start + k * y_vector, x_start + k * x_vector

def get_around_points_coarse(point, radius):
    """
    返回围绕节点的方形区域中的坐标点
    :return:
    """
    z, y, x = point
    r = math.ceil(radius)
    x = math.floor(x)
    y = math.floor(y)
    z = math.floor(z)
    return [(zz, yy, xx) for xx in range(x - r, x + r + 1)
                         for yy in range(y - r, y + r + 1)
                         for zz in range(z - r, z + r + 1)]

def get_around_points(point, radius, resolution):
    """
    获取以节点坐标为中心点，self.radius为半径内的左右坐标点
    :param r_offset: 有些情况下要对该节点的半径值增加一个缓冲值
    :return:
    """
    z, y, x = point
    coordinates_init = get_around_points_coarse(point = point, radius = radius)
    radius_basic_z = radius * resolution[0]
    radius_basic_y = radius * resolution[1]
    radius_basic_x = radius * resolution[2]
    assert radius_basic_z != 0, '{} = 0, radius = {}, resolution = {}'.format(radius_basic_z, radius, resolution)
    assert radius_basic_y != 0, '{} = 0, radius = {}, resolution = {}'.format(radius_basic_y, radius, resolution)
    assert radius_basic_x != 0, '{} = 0, radius = {}, resolution = {}'.format(radius_basic_x, radius, resolution)
    coordinates = [(zz, yy, xx)
                   for (zz, yy, xx) in coordinates_init
                   if ((zz - z) / radius_basic_z) ** 2 +
                      ((yy - y) / radius_basic_y) ** 2 +
                      ((xx - x) / radius_basic_x) ** 2 <= 1]
    return coordinates




if __name__ == '__main__':
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    point0 = (0,0,0)
    point1 = (0,0,5)
    the_coor = get_connect_coordinate_3d_new(point0, point1, radius = 3)
    x_list = [t[0] for t in the_coor]
    y_list = [t[1] for t in the_coor]
    z_list = [t[2] for t in the_coor]
    ax = plt.subplot(111, projection = '3d')
    ax.scatter(x_list, y_list, z_list, c = 'r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    
    point_start = (0,0,0)
    vector = (1, 3, 4)
    distance = 5
    print(get_coordinate_along_vector(point_start, vector, distance))
    
    """

    neuron_node = get_around_points((0, 0, 0), resolution = (1, 1, 1), radius = 1)
    print(neuron_node)