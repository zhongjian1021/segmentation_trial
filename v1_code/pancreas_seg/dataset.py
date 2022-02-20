from torchvision.datasets.mnist import read_image_file, read_label_file
from utils import tensor2numpy, show2D, get_filelist_frompath, list_unique, readimg, list_slide
import os
import csv
import PIL
sep = os.sep
from config_dataset import config as data_config
from wama.utils import *
import SimpleITK as sitk







DATASET_NAME = list(r'MNIST,CIFAR10,CIFAR100,imagenette,imagewoof'.split(','))
DATASET_NAME.append('miccai2018pancreas')
DATASET_NAME.append('NIH_pancreas')



# 打印一些数据及相关的东西
print('current support datasets:', DATASET_NAME)
print('imagenette 和 imagewoof 数据集部分图像为单通道的， \n这些图像会被统一处理（叠加）为三通道')


# 简单数据集（分辨率比较低，32或64级别）
MNIST_data_root = data_config['MNIST_data_root']
CIFAR10_data_root = data_config['CIFAR10_data_root']
CIFAR100_data_root = data_config['CIFAR100_data_root']
# medical_minist数据集可以一试
# todo
# imagenet子集有两个size的，160和320（只是大概尺寸，图片非方形）
imagenette160_data_root = data_config['imagenette160_data_root']
imagenette320_data_root = data_config['imagenette320_data_root']
imagewoof160_data_root = data_config['imagewoof160_data_root']
imagewoof320_data_root = data_config['imagewoof320_data_root']

# 医学数据集-2D
# 超声数据集，X光数据集




# 医学数据集-3D
# 胰腺（具体名字待补充，CT 的），
miccai_2018_decathlon_data_root = data_config['miccai_2018_decathlon_data_root']
NIH_pancreas_data_root = data_config['NIH_pancreas_data_root']


# BraTS 2018（脑胶质瘤的， MRI的），
# LiTS 20？？ 洪源哥的，CT的肝脏的


# imagenet全部数据集（这个最后再搞）












# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
def get_dataset(d_name, **kwargs):
    """
    总的获取数据集的函数
    :param
    d_name:
        自然图像数据集：MNIST, CIFAR10, CIFAR100, imagenette, imagewoof  |
        医学数据集：LiTS、NIH_pancreas,
    :return:
    """
    # print(bool(kwargs))

    d_name_list =  list(r'MNIST,CIFAR10,CIFAR100,imagenette,imagewoof'.split(','))
    if d_name not in d_name_list:
        raise ValueError('only support:'+ str(d_name_list))

    get_dataset_func = [get_dataset_MNIST,
                        get_dataset_CIFAR10,
                        get_dataset_CIFAR100,
                        get_dataset_imagenette,
                        get_dataset_imagewoof,
                        ]
    try:
        if kwargs:
            return get_dataset_func[d_name_list.index(d_name)](**kwargs)
        else:
            return get_dataset_func[d_name_list.index(d_name)]()
    except:
        print('load failed')
        raise ValueError

# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
# MNIST =============================================================================================
def get_dataset_MNIST():

    train_image = MNIST_data_root+sep+r'train-images-idx3-ubyte\train-images.idx3-ubyte'
    train_label = MNIST_data_root+sep+r'train-labels-idx1-ubyte\train-labels.idx1-ubyte'
    test_image = MNIST_data_root+sep+r't10k-images-idx3-ubyte\t10k-images.idx3-ubyte'
    test_label = MNIST_data_root+sep+r't10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte'


    training_set = [tensor2numpy(read_image_file(train_image)), tensor2numpy(read_label_file(train_label))]
    testing_set = [tensor2numpy(read_image_file(test_image)), tensor2numpy(read_label_file(test_label))]

    # 制作数据集，数据集是一个list，每个element都是一个dict
    # dict包含的字段，暂时规定为，img 图像，label ，label_name， img_path 图像所在的路径，这个没有可以直接写None
    train_list = [dict(img = training_set[0][i],
                       img_path = '',
                       label = training_set[1][i],
                       label_name = str(training_set[1][i]))
                  for i in range(len(training_set[1]))]
    test_list = [dict(img = testing_set[0][i],
                       img_path = '',
                       label = testing_set[1][i],
                       label_name = str(testing_set[1][i]))
                  for i in range(len(testing_set[1]))]

    return dict(train_set = train_list,
                test_set = test_list,
                train_set_num = len(train_list),
                test_set_num = len(test_list),
                dataset_name = 'MNIST')



# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
# CIFAR10 =============================================================================================
import numpy as np
from six.moves import cPickle as pickle
import os
import platform

# 读取文件
def load_pickle(f):
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == '2':
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return [X, Y]

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return [Xtr, Ytr, Xte, Yte]

def get_dataset_CIFAR10():
    Xtr, Ytr, Xte, Yte = load_CIFAR10(CIFAR10_data_root)
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_list = [dict(img = Xtr[i],
                       img_path = '',
                       label = Ytr[i],
                       label_name = class_name[Ytr[i]])
                  for i in range(len(Ytr))]
    test_list = [dict(img = Xte[i],
                       img_path = '',
                       label = Yte[i],
                       label_name = class_name[Yte[i]])
                  for i in range(len(Yte))]

    return dict(train_set = train_list,
                test_set = test_list,
                train_set_num = len(train_list),
                test_set_num = len(test_list),
                dataset_name='CIFAR10')

def unpickleCifar100(file):
    import pickle
    fo = open(file, 'rb')
    #dict = pickle.load(fo)#会报错，改为如下
    dict = pickle.load(fo,encoding='iso-8859-1')
    fo.close()
    return dict

def get_dataset_CIFAR100():
    # 注意，100和10不一样，有细粒度类别和粗类别
    train_path = CIFAR100_data_root+sep+r'train'
    test_path = CIFAR100_data_root+sep+r'test'
    meta_path = CIFAR100_data_root+sep+r'meta'
    train_dict = unpickleCifar100(train_path)
    test_dict = unpickleCifar100(test_path)
    meta_dict = unpickleCifar100(meta_path)

    # 改变一些形状
    train_dict['data'] = train_dict['data'].reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    test_dict['data'] = test_dict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")


    # 变成字典形式
    train_list = [dict(img = train_dict['data'][i],
                       img_path = '',
                       coarse_label = train_dict['coarse_labels'][i],  # 粗标签  20类别
                       fine_label = train_dict['fine_labels'][i],  # 细标签 100类别
                       coarse_label_name = meta_dict['coarse_label_names'][train_dict['coarse_labels'][i]],
                       fine_label_name = meta_dict['fine_label_names'][train_dict['fine_labels'][i]])
                  for i in range(len(train_dict['fine_labels']))]

    test_list = [dict(img = test_dict['data'][i],
                       img_path = '',
                       coarse_label = test_dict['coarse_labels'][i],  # 粗标签  20类别
                       fine_label = test_dict['fine_labels'][i],  # 细标签 100类别
                       coarse_label_name = meta_dict['coarse_label_names'][test_dict['coarse_labels'][i]],
                       fine_label_name = meta_dict['fine_label_names'][test_dict['fine_labels'][i]])
                  for i in range(len(test_dict['fine_labels']))]



    return dict(train_set = train_list,
                test_set = test_list,
                train_set_num = len(train_list),
                test_set_num = len(test_list),
                dataset_name = 'CIFAR100')


# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================
# imagenet subset=============================================================================================

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def get_dataset_imagenet_sub(root_path, preload_cache = False, aim_shape = None, order = 0, label_level = 0):
    """

    :param root_path:
    :param preload_cache: 是否提前加载到内存
    :param aim_shape:reshape的形状，256、[256]，[256,256]三种格式都可以
    :param order: reshape的阶数，0最近邻，3cubic
    :param label_level: 标签的噪声比例，只能选[0、1、5、25、50]中的一个
    :return:
    """

    # 检查label_level是否在[0、1、5、25、50]里
    if label_level not in [0,1,5,25,50]:
        raise ValueError('label_level have to be in [0,1,5,25,50]')
    label_index = [0,1,5,25,50].index(label_level)


    # 首先根据csv文件读取类别和路径信息
    csv_file = get_filelist_frompath(root_path, 'csv')[0]
    CTcsvlines = readCsv(csv_file)
    header = CTcsvlines[0]
    # print('header', header)
    images_labels = CTcsvlines[1:]

    all_cate = list_unique([i[1] for i in images_labels])
    all_cate.sort()  # 排序保证每一次的cate 序号都一样

    train_list = []
    test_list = []
    for case in images_labels:
        if case[-1] == 'False':  # is_valid
            train_list.append(dict(
                img=None,
                img_path=root_path+sep+case[0],
                label=all_cate.index(case[label_index+1]),
                label_name=case[label_index+1] ))
        else:
            test_list.append(dict(
                img=None,
                img_path=root_path + sep + case[0],
                label=all_cate.index(case[label_index + 1]),
                label_name=case[label_index + 1]))


    if preload_cache:
        print('loading into cache')
        for _item in train_list:
            _item['img'] = readimg(_item['img_path'], aim_shape, order)
        for _item in test_list:
            _item['img'] = readimg(_item['img_path'], aim_shape, order)

    return dict(train_set = train_list,
                 test_set = test_list,
                 train_set_num = len(train_list),
                 test_set_num = len(test_list))

def get_dataset_imagenette(orsize = 160, preload_cache = False, aim_shape = None, order = 0, label_level = 0):
    if orsize == 160:
        root_path = imagenette160_data_root
    elif orsize == 160:
        root_path = imagenette320_data_root
    else:
        raise ValueError('only 160 or 320')
    return_dict = get_dataset_imagenet_sub(root_path, preload_cache, aim_shape, order, label_level)
    return_dict.update({'dataset_name': 'imagenette'})
    return return_dict

def get_dataset_imagewoof(orsize = 160, preload_cache = False, aim_shape = None, order = 0, label_level = 0):
    if orsize == 160:
        root_path = imagewoof160_data_root
    elif orsize == 160:
        root_path = imagewoof320_data_root
    else:
        raise ValueError('only 160 or 320')
    return_dict = get_dataset_imagenet_sub(root_path, preload_cache, aim_shape, order, label_level)
    return_dict.update({'dataset_name':'imagewoof'})
    return return_dict


def from_pth2array(path, dataset_name, order = 2, aim_shape = None):
    """
    针对不同数据集，有不同的读取方法，不支持mnist和cifar系列！
    # 目前只支持imagenette和imagewoof
    :param path:
    :param dataset_name:
    :return:
    """

    # 检查dataset_name是否有误
    if dataset_name == r'imagenette' or dataset_name == r'imagewoof':
        img = readimg(path, aim_shape, order)
    else:
        raise ValueError

    return img


# pancreas subset=============================================================================================
# pancreas subset=============================================================================================
# pancreas subset=============================================================================================
# pancreas subset=============================================================================================
# pancreas subset=============================================================================================
# pancreas subset=============================================================================================
# pancreas subset=============================================================================================
# 胰腺的分割数据集，暂时包括两个，其中miccai10项全能的是2类分割（有肿瘤和胰腺，所以需要根据参数决定是否之保留胰腺label）
# 2018 10项目全能的是门脉期CT， NIH的看起来也是门脉期
# 预处理的时候，会尽量保证两个数据集图像相似，所以去除背景这一步骤是必要的



def resize4pancreasNII(scan, aimspace = [128, 128, 64], order = 0, is_mask = False, is_NIH = False):
    """
    # 由于医学图像的特性，xy分辨率较高，所以首先对image x和y进行resize，z等比例保持不变
    # 1）如果z稍微大于这目标z，则resize到目标z ； 如果z大于z太多，则需要把上面的减裁掉（因为胰腺在下面，所以把肺部减去），减到符合标准再resize
    # 2）如果z略微小于目标，则同样resize到目标z； 否则在上面补0，补到符合标准，再resize
    # aim_size 这个要根据resample后的总体尺寸来定，不要瞎搞，[128,128,64]就差不多（对于这个数据集）

    """
    # scan = image_reader.scan['CT']
    # 保持z轴相对比例,先将xy缩放到对应shape
    scan = resize3D(scan, aimspace[:2]+[(scan.shape[-1]/scan.shape[0])*aimspace[0]], order)


    # 注意，这里以miccai2018 的为标准，调整nii的数据使之方向和miccai一样
    # 也就是需要下面这个操作
    # show3Dslice(scan[:,:,22:])
    # show3Dslice(scan)
    # show3Dslice(scan_NIH[::-1,::-1,::-1])
    # show3Dslice(scan_miccai)
    if is_NIH:
        scan = scan[::-1,::-1,::-1]


    if True:
        thresold = (5/64)*aimspace[-1] # todo 自己设定的阈值，64基础上，上下可以差4
        if abs(scan.shape[-1] - aimspace[-1]) <= thresold:
            # 如果z很接近，就直接resize z轴到目标尺寸
            scan = resize3D(scan,aimspace, order)
        elif abs(scan.shape[-1] - aimspace[-1]) > thresold and (scan.shape[-1] - aimspace[-1])>=0:
            # 如果层数过多，则删除底部（胯部）到阈值+aimspace(因为胰腺一般靠近肝脏和肺部，而不靠近跨部），之后再resize
            # cut_slices = scan.shape[-1] -
            # scan = scan[:,:,:int((aimspace[-1]+thresold))]  # 注意这个顺序 todo 有点问题 mmp，部分label会被切掉，暂时不要这个操作
            scan = resize3D(scan,aimspace, order)
        elif abs(scan.shape[-1] - aimspace[-1]) > thresold and (scan.shape[-1] - aimspace[-1])<0:
            # 如果层数过多，则在顶部（肺部）添加0层到阈值-aimspace，之后再resize
            cut_slices = abs(scan.shape[-1] - (aimspace[-1]-thresold))
            tmp_scan = np.zeros(aimspace[:2]+[int(scan.shape[-1]+cut_slices)], dtype=scan.dtype)
            if is_mask:
                pass # 如果是分割mask，则赋予0
            else:
                tmp_scan = tmp_scan - 1024 # 如果是CT，则赋予空气值
            tmp_scan[:,:,:scan.shape[-1]] = scan
            scan = resize3D(scan,aimspace, order)
        else:
            scan = resize3D(scan, aimspace, order)

    return scan

def remove_bg4pancreasNII(scan):
    """
    # 此外，CT图像预处理，可以包含“去除非身体的扫描床部分”
    # 也就是去除无关地方，这可以极大减少冗余的地方
    # 这个正好也可以在袁总的数据上用到
    # 一般来说，CT值小于-850 的地方，就可以不要了，不过还是要留一个参数控制阈值
    # 思路：二值化，开操作，取最大联通，外扩，剩下的都不要，完事，记得也要把截取矩阵输出，以供分割使用

    :param scan: 没有卡过窗宽窗外的图像！！！！
    :return:
    """


    # scan = image_reader.scan['CT']
    # scan = resize3D(scan,[256,256,None])
    # show3D(scan)
    # show3D(scan_mask_af)

    scan_mask = (scan> -900).astype(np.int)
    sitk_img = sitk.GetImageFromArray(scan_mask)
    sitk_img = sitk.BinaryMorphologicalOpening(sitk_img!=0, 15)
    scan_mask_af = sitk.GetArrayFromImage(sitk_img)
    # show3Dslice(np.concatenate([scan_mask_af, scan_mask],axis=1))
    scan_mask_af = connected_domain_3D(scan_mask_af)
    # 计算得到bbox，形式为[dim0min, dim0max, dim1min, dim1max, dim2min, dim2max]
    indexx = np.where(scan_mask_af > 0.)
    dim0min, dim0max, dim1min, dim1max, dim2min, dim2max = [np.min(indexx[0]), np.max(indexx[0]),
                                                            np.min(indexx[1]), np.max(indexx[1]),
                                                            np.min(indexx[2]), np.max(indexx[2])]
    return [dim0min, dim0max, dim1min, dim1max]

def read_nii2array4miccai_pancreas(img_pth, mask_pth, aim_spacing, aim_shape, order=3, is_NIH = False, cut_bg=True):
    """

    :param img_pth:
    :param mask_pth:
    :param aim_spacing:
    :param aim_shape:
    :param order:
    :param is_NIH: 如果是NIH数据集，则需要调整各个维度顺序，使之和MICCAI一样
    :return:
    """
    # img_pth = case['img_path']
    # mask_pth = case['mask_path']
    # aim_shape = [128,128,64]
    # aim_spacing = [0.5,0.5,0.8]


    image_reader = wama()  # 构建实例
    image_reader.appendImageAndSementicMaskFromNifti('CT', img_pth, mask_pth)

    # # 修正label(原始数据是错的，一定要先修正，如果使用的是经过修正的，就算了）
    # if is_NIH:
    #     image_reader.sementic_mask['CT'] = image_reader.sementic_mask['CT'][::-1,:,:]


    # (不要在这里调整窗宽窗位，因为可能用到多窗宽窗位）
    # image_reader.adjst_Window('CT', 321, 123)
    # resample
    if aim_spacing is not None:
        print('resampling to ', aim_spacing, 'mm')
        image_reader.resample('CT', aim_spacing, order=order)  # 首先resample没得跑,[0.5,0.5,0.8]就好

    # 去除多余部分
    # scan = image_reader.scan['CT']
    if cut_bg:
        print('cuting bg')
        bbox = remove_bg4pancreasNII(image_reader.scan['CT'])
        image_reader.scan['CT'] = image_reader.scan['CT'][bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        image_reader.sementic_mask['CT'] = image_reader.sementic_mask['CT'][bbox[0]:bbox[1], bbox[2]:bbox[3], :]


    # resize到固定大小
    scan = resize4pancreasNII(image_reader.scan['CT'], aimspace=aim_shape, order=order, is_mask=False, is_NIH= is_NIH)
    mask = resize4pancreasNII(image_reader.sementic_mask['CT'], aimspace=aim_shape, order=0, is_mask=True, is_NIH=is_NIH)  # 注意mask是order 0

    # 由于mask存在肿瘤和胰腺，而我们只需保存胰腺即可，所以要把胰腺和肿瘤合并！
    mask = (mask >= 0.5).astype(mask.dtype)


    return scan,mask

def get_dataset_miccai2018pancreas(preload_cache = False, order = 3):
    print('getting miccai2018pancreas')
    dataset_name = r'miccai2018pancreas'
    root = data_config['miccai_2018_decathlon_data_root']
    aim_spacing = data_config['miccai_2018_decathlon_data_aimspace']
    aim_shape = data_config['miccai_2018_decathlon_data_aimshape']
    cut_bg = data_config['miccai_2018_decathlon_data_cut_bg']

    nii_list =get_filelist_frompath(root+sep+r'imagesTr', 'nii.gz')

    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for case in nii_list:
            train_list.append(dict(
                img=None,
                mask =None,
                img_path=case,
                mask_path=root+sep+'labelsTr'+sep+case.split(sep)[-1],
                ))

    # train_list = train_list[25:28]


    # 预先读取数据
    if preload_cache:
        print('loading.')
        for index, case in enumerate(train_list):
            print('loading ',index+1,'/', len(train_list),'...')
            scan, mask = read_nii2array4miccai_pancreas(case['img_path'], case['mask_path'],aim_spacing, aim_shape, order, False, cut_bg)
            print(scan.shape)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)



    return dict(train_set = train_list,
                 train_set_num = len(train_list),
                dataset_name = dataset_name)

def get_dataset_NIH_pancreas(preload_cache = False, order = 3):
    print('getting NIH_pancreas')
    dataset_name = r'NIH_pancreas'
    root = data_config['NIH_pancreas_data_root']
    aim_spacing = data_config['NIH_pancreas_data_aimspace']
    aim_shape = data_config['NIH_pancreas_data_aimshape']
    cut_bg = data_config['NIH_pancreas_data_cut_bg']

    nii_list =get_filelist_frompath(root+sep+r'data', 'nii.gz')

    # 先把path装进去(没有测试集，只有训练集）
    train_list = []
    for case in nii_list:
            train_list.append(dict(
                img=None,
                mask =None,
                img_path=case,
                mask_path=root+sep+'mask'+sep+'label'+case.split(sep)[-1][-11:],
                ))

    # train_list = train_list[1:2]


    # 预先读取数据
    if preload_cache:
        print('loading.')
        for index, case in enumerate(train_list):
            print('loading ',index+1,'/', len(train_list),'...')
            scan, mask = read_nii2array4miccai_pancreas(case['img_path'], case['mask_path'],aim_spacing, aim_shape, order, True, cut_bg)
            print(scan.shape)
            case['img'] = scan.astype(np.float32)
            case['mask'] = mask.astype(np.uint8)


    return dict(train_set = train_list,
                 train_set_num = len(train_list),
                dataset_name = dataset_name)

def make_dataset_2D(dataset, remove_none_slice = True):
    """
    仅仅适用与NIH和miccai胰腺数据，其他数据需要改代码
    :param dataset:
    :param remove_none_slice: 是否去除多余层
    :return:
    """

    # dataset = dataset_test
    print('making 2D')

    train_list = []
    # 第几层、属于哪张图片，要保留
    for ii, case in enumerate(dataset['train_set']):
        print(ii+1, '/', dataset['train_set_num'])
        # 计算需要保留的层的范围
        if remove_none_slice:
            indexx = np.where(case['mask'] > 0.01)
            slice_index_min, slice_index_max = [np.min(indexx[2]), np.max(indexx[2])]
        else:
            slice_index_min, slice_index_max = [0, case['mask'].shape[-1]]

        for slice_index in range(slice_index_min,slice_index_max,1):
            train_list.append(
                dict(
                    img = case['img'][:,:,slice_index],
                    mask = case['mask'][:,:,slice_index],
                    img_path = case['img_path'],
                    mask_path = case['mask_path'],
                    slice_index = slice_index,
                )
            )

    return dict(train_set = train_list,
                 train_set_num = len(train_list),
                 dataset_name = dataset['dataset_name'])

















if __name__ == '__main__':
    # dataset= get_dataset_MNIST()
    # dataset= get_dataset_miccai2018pancreas(preload_cache=True, order=0)
    # save_as_pkl(r'/media/szu/2.0TB_2/wmy/@database/miccai_2018_decathlon/Pancreas Tumour_precessed/pre_order0_128_128_64_new.pkl',
    #             [dataset, data_config])

    dataset2= get_dataset_NIH_pancreas(preload_cache=True, order=0)
    save_as_pkl(r'/media/szu/2.0TB_2/wmy/@database/NIH-pancreas/Pancreas-CT/pre_order0_256_256_128_new.pkl',
                [dataset2,data_config] )

    # 可以直接变成2D的数据集
    # dataset_test2D = make_dataset_2D(dataset_test, False)
    # show2D(dataset_test2D['train_set'][6]['img'])



    # 检查mask和原图是否匹配
    # show3Dslice(np.concatenate([mat2gray(dataset['train_set'][0]['img']),0.5*mat2gray(dataset['train_set'][0]['img'])+0.5*dataset['train_set'][0]['mask']],axis=1))
    # show3Dslice(np.concatenate([mat2gray(dataset['train_set'][0]['img']),dataset['train_set'][0]['mask']],axis=1))
    # show2D(mat2gray(dataset['train_set'][0]['img'][:,:,32]))
    # show2D(mat2gray(dataset['train_set'][0]['mask'][:,:,32]))
	#
    # show3Dslice(np.concatenate([mat2gray(dataset2['train_set'][0]['img']),0.5*mat2gray(dataset2['train_set'][0]['img'])+0.5*dataset2['train_set'][0]['mask']],axis=1))
    # show3Dslice(np.concatenate([mat2gray(dataset2['train_set'][0]['img']), dataset2['train_set'][0]['mask']], axis=1))
    # show2D(mat2gray(dataset2['train_set'][0]['img'][:, :, 32]))
    # show2D(mat2gray(dataset2['train_set'][0]['mask'][:, :, 32]))
	#
    # # 检查mask是不是二值，以及方向是不是一致
    # show3D((dataset['train_set'][2]['mask']))
    # show3D((dataset2['train_set'][0]['mask']))
	#
    # # 检查原图是不是方向一致
    # # 要求，mayavi显示后，肝脏在右，肺部在上
    # show3D((dataset['train_set'][2]['img']))
    # show3Dslice((dataset2['train_set'][2]['img']))


    # save_as_pkl(r'F:\9database\database\miccai_2018_decathlon\Pancreas Tumour_precessed\pre_order3_128_128_64.pkl',[dataset,data_config] )
    # save_as_pkl(r'F:\9database\database\NIH-pancreas\Pancreas-CT\pre_order3_128_128_64.pkl',[dataset2,data_config] )
	#
	#
	#
    # dataset_pre = load_from_pkl(r'F:\9database\database\miccai_2018_decathlon\Pancreas Tumour_precessed\pre_order0_128_128_64.pkl')
    # # show3Dslice(np.concatenate([dataset_pre['train_set'][0]['img'],dataset['train_set'][0]['mask']],axis=1))
    # dataset = get_dataset_CIFAR10()
    # dataset = get_dataset_CIFAR100()
    # dataset = get_dataset_imagewoof()
    # dataset = get_dataset_imagenette(preload_cache=True)
    # dataset = get_dataset_imagewoof(preload_cache=True, aim_shape=32)
    # for _ in range(9):
    # #     dataset = list_slide(dataset, 3)
    #     dataset['train_set'] = list_slide(dataset['train_set'], 3)
    #     # dataset.update({'dataset_name':'imagewoof'})
    #     show2D(dataset['train_set'][0]['img'] / 255)






