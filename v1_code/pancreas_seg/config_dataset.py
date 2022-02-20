# dataset 的设置
# 比如各个3D数据集的预处理参数等等



config = dict(





# todo 数据集路径设置
# 简单数据集（分辨率比较低，32或64级别）
MNIST_data_root = r'F:\9database\database\MNIST\MNIST_data\MNIST_data',
CIFAR10_data_root = r'F:\9database\database\CIFAR-10\cifar-10-python.tar\cifar-10-python\cifar-10-batches-py',
CIFAR100_data_root = r'F:\9database\database\CIFAR-100\cifar-100-python.tar\cifar-100-python\cifar-100-python',

# medical_minist数据集可以一试



# imagenet子集有两个size的，160和320（只是大概尺寸，图片非方形）
imagenette160_data_root = r'F:\9database\database\imagenet_subset\imagenette2-160\imagenette2-160\imagenette2-160',
imagenette320_data_root = r'F:\9database\database\imagenet_subset\imagenette2-320\imagenette2-320\imagenette2-320',
imagewoof160_data_root = r'F:\9database\database\imagenet_subset\imagewoof2-160\imagewoof2-160\imagewoof2-160',
imagewoof320_data_root = r'F:\9database\database\imagenet_subset\imagewoof2-320\imagewoof2-320\imagewoof2-320',

# 医学数据集-2D
# 超声数据集，X光数据集




# 医学数据集-3D
# 胰腺（具体名字待补充，CT 的），
miccai_2018_decathlon_data_root = r'/media/szu/2.0TB_2/wmy/@database/miccai_2018_decathlon/Pancreas Tumour/Task07_Pancreas/Task07_Pancreas',
miccai_2018_decathlon_data_WW = 321, # 窗宽
miccai_2018_decathlon_data_WL = 123, # 窗位
miccai_2018_decathlon_data_aimspace = [0.5,0.5,0.8], # respacing
# miccai_2018_decathlon_data_aimspace = [1.0,1.0,1.6], # respacing
# miccai_2018_decathlon_data_aimspace = None, # respacing
miccai_2018_decathlon_data_aimshape = [128,128,64], # 最终形状，经过resize和减裁的,todo 这个要自己好好计算,目前我计算的比例就是[1.0,1.0,1.6]对应[128,128,64]
# miccai_2018_decathlon_data_aimshape = [96,96,48], # 最终形状，经过resize和减裁的,todo 这个要自己好好计算,目前我计算的比例就是[1.0,1.0,1.6]对应[128,128,64]
miccai_2018_decathlon_data_cut_bg =False, # 去掉背景 todo 这个步骤及其消耗时间

NIH_pancreas_data_root = r'/media/szu/2.0TB_2/wmy/@database/NIH-pancreas/Pancreas-CT',
NIH_pancreas_data_WW = 321, # 窗宽
NIH_pancreas_data_WL = 123, # 窗位
NIH_pancreas_data_aimspace = [0.5,0.5,0.8], # respacing
# NIH_pancreas_data_aimspace = [1.0,1.0,1.6], # respacing
# NIH_pancreas_data_aimspace = None, # respacing
NIH_pancreas_data_aimshape = [256,256,128], # 最终形状，经过resize和减裁的,todo 这个要自己好好计算
# NIH_pancreas_data_aimshape = [96,96,48], # 最终形状，经过resize和减裁的,todo 这个要自己好好计算
NIH_pancreas_data_cut_bg =False, # 去掉背景

# BraTS 2018（脑胶质瘤的， MRI的），
# LiTS 20？？ 洪源哥的，CT的肝脏的


# imagenet全部数据集（这个最后再搞）


)