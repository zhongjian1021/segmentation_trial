"""
 应该搞一个高级接口，形如
network_wait4pretrained = resnet50()  # 要有固定的输出形式，一般就是多层特征，最多加一个后面的fc层
MiS_model = MiS(model = network_wait4pretrained, arg1 = xx, ...)
计算loss可以学习检测框架的架构，放在网络里面实现
1)最好可以自动获得输入的形状之类的



"""
import torch.nn as nn
import torch
import torch.nn.functional as F
norm_set = 'gn'
gn_c = 8


class fake_backbone(nn.Module):
	# todo 待完成，先搞Unet再说，这个很大几率和Unet一样
	# 但是都是针对分割网络
	# 如果是分类任务or检测任务
	# 如何更好的监督编码器？ 一定需要解码器结构么
	# 比如simclr或者moco或者byol，其实解码器就相当于一个比较复杂的g(x)
	# 指的探讨
	def __init__(self):
		super().__init__()
	def forward(self, x):
		b = torch.ones([3,64,220,120]) # 下面这些，b开始的，都对应
		c = torch.ones([3,128,154,54])
		d = torch.ones([3,256,132,32])
		e = torch.ones([3,512,116,16])
		f = torch.ones([3,1024,18,8])

		return [b,c,d,e,f]

class fake_backbone3D(nn.Module):
	# todo 待完成，先搞Unet再说，这个很大几率和Unet一样
	# 但是都是针对分割网络
	# 如果是分类任务or检测任务
	# 如何更好的监督编码器？ 一定需要解码器结构么
	# 比如simclr或者moco或者byol，其实解码器就相当于一个比较复杂的g(x)
	# 指的探讨
	def __init__(self):
		super().__init__()
	def forward(self, x):
		b = torch.ones([3,2,32,32,32]) # 下面这些，b开始的，都对应
		c = torch.ones([3,3,28,28,28])
		d = torch.ones([3,4,16,16,16])
		e = torch.ones([3,5,8,8,8])
		f = torch.ones([3,6,4,4,4])

		return [b,c,d,e,f]

class MiS(nn.Module):
	# todo 待完成，先搞Unet再说，这个很大几率和Unet一样
	# 但是都是针对分割网络
	# 如果是分类任务or检测任务
	# 如何更好的监督编码器？ 一定需要解码器结构么
	# 比如simclr或者moco或者byol，其实解码器就相当于一个比较复杂的g(x)
	# 指的探讨
	def __init__(self, model):
		super().__init__()
		self.model = model

class Unet_beta(nn.Module):
	"""
	（考虑增加注意力or GATE机制）
	注意，这个Unet只构建解码器，编码器需要额外搞，所以只有上采样的conv模块
	1）可以控制skip connection，比如只在1和3有skip connection
	2）可以控制上采样后卷积输出通道
	3）输入可以是任意形状、通道、数量的特征图，是多尺度，如resnet的多尺度特征
	4）还要控制输出，比如输入5个尺度，输出其实可以只有3个尺度，这个其实可以直接深监督的套路
	5）控制上采样的形状变化，建议统一采用unsample执行
	6）下采样使用何种池化，也要可以指定才行，暂定三种，avg，mean，mean_avg(不过因为本decoder只实现上采样，所以不考虑这个）
	"""
	def __init__(self,
				 output_channel,
				 backbone,
				 input_feature_channel=[64,   128,   256,   512],  # 最后一个特征图开始上采样
				 skip_connection =     [True, True, True],
				 middle_conv_channel = [64,   128,   256],
				 final_conv_channel =  [64,   128,   256],
				 ):

		# 上采样，用尺寸自适应那个套路来搞，每次update一下shape相关的操作即可
		super().__init__()

		"""
		首先检查输入参数是否有误
		需要满足 len(input_feature_channel) = len(skip_connection) +1
		                                   = len(middle_conv_channel) +1
		                                   = len(final_conv_channel) +1
		"""
		print("""
				首先检查输入参数是否有误
				需要满足 len(input_feature_channel) = len(skip_connection) +1
		                                   		= len(middle_conv_channel) +1
		                                   		= len(final_conv_channel) +1""")

		if ((len(input_feature_channel) != len(skip_connection) +1) or
			(len(input_feature_channel) != len(middle_conv_channel) +1) or
			(len(input_feature_channel) != len(final_conv_channel) +1)):
			raise ValueError


		# 首先记录初始化的各个控制参数
		self.input_channel = output_channel
		self.input_feature_channel = input_feature_channel
		self.skip_connection = skip_connection
		self.middle_conv_channel = middle_conv_channel
		self.final_conv_channel = final_conv_channel



		# 初始化网络层（带参数层）---------------------------------------------------------
		self.backbone = backbone


		conv_modules_list = []
		# 首先添加最下层（因为输入为input的2个，因此单独添加
		conv_modules_list.append(
			self.make_conv_module(
				input_c=self.input_feature_channel[-1],
				skip_c=self.input_feature_channel[-2],
				middle_c=self.middle_conv_channel[-1],
				final_c=self.final_conv_channel[-1],
				skip_connection_flag=self.skip_connection[-1]
			))

		for index in range(len(self.skip_connection) - 1):
			conv_modules_list.append(
				self.make_conv_module(
					input_c=self.final_conv_channel[-(index + 1)],
					skip_c=self.input_feature_channel[-(index + 3)],
					middle_c=self.middle_conv_channel[-(index + 2)],
					final_c=self.final_conv_channel[-(index + 2)],
					skip_connection_flag=self.skip_connection[-(index + 2)]
				))


		self.conv_modules_list = nn.ModuleList(conv_modules_list) # 顺序是，从深层特征 conv到浅层的，即尺寸从小到大


		# 最后分割的head
		self.seg_head = nn.Conv2d(self.final_conv_channel[0], output_channel, kernel_size=1)

		# 初始化网络层（非参数层，即上采样层）--------------------------------------------------------
		# 上采样块数量n=len(input_feature_channel)
		self.upsample_list = nn.ModuleList([nn.UpsamplingBilinear2d(size=(2, 2))
											for _ in range(len(input_feature_channel))])

	def make_conv_module(self, input_c, skip_c, middle_c, final_c, skip_connection_flag):
		if skip_connection_flag:
			input_c = input_c + skip_c
		return VGGBlock(input_c, middle_c, final_c)

	def update_upsample_size(self, x):
		# x 是[ input, feature1, feature2, ...]这种形式
		for index in range(len(x)-1):
			self.upsample_list[index] = nn.UpsamplingBilinear2d(size=(x[-(index+2)].shape[2], x[-(index+2)].shape[3]))
			# 顺序是从深层特征到浅层(倒数第二个特征一直到input），因为要和conv list保持一致

	def forward(self, x):
		# 将输入储存到list
		input = [x]

		# 获取层级特征
		backboneF = self.backbone(x)

		# 将输入和层级特征的两个list合并
		x = input + backboneF  # 即[ input, feature1, feature2, ...]这种形式

		# 这里一步到位，一次性调整所有上采样模块
		self.update_upsample_size(x)

		# 开始传火，首先对最下层特征操作
		reture_f_list = []
		if self.skip_connection[-1]:
			up_f = self.conv_modules_list[0](
				torch.cat([x[-2], self.upsample_list[0](x[-1])], axis=1)
			)
		else:
			up_f = self.conv_modules_list[0](
				self.upsample_list[0](x[-1])
			)
		reture_f_list.append(up_f)


		# 之后对其余层特征操作
		for index in range(len(self.skip_connection)-1):
			if self.skip_connection[-(index+2)]:
				up_f = self.conv_modules_list[index+1](
					torch.cat([x[-(index+3)], self.upsample_list[index+1](up_f)], axis=1)
				)
			else:
				up_f = self.conv_modules_list[index+1](
					self.upsample_list[index + 1](up_f)
				)
			reture_f_list.append(up_f)

		# 最后计算得到输出map
		final_out = self.seg_head(self.upsample_list[-1](up_f))


		# 返回各个尺寸的特征图（供debug用）
		return dict(segout = final_out, f_list = reture_f_list)

class Unet_beta3D(nn.Module):
	"""
		Unet_beta 3D 版本 todo 需要好好修改一下
	"""
	def __init__(self,
				 output_channel,
				 backbone,
				 input_feature_channel=[64,   128,   256,   512],  # 最后一个特征图开始上采样
				 skip_connection =     [True, True, True],
				 middle_conv_channel = [64,   128,   256],
				 final_conv_channel =  [64,   128,   256],
				 ):

		super().__init__()

		print("""
				首先检查输入参数是否有误
				需要满足 len(input_feature_channel) = len(skip_connection) +1
		                                   		= len(middle_conv_channel) +1
		                                   		= len(final_conv_channel) +1""")

		if ((len(input_feature_channel) != len(skip_connection) +1) or
			(len(input_feature_channel) != len(middle_conv_channel) +1) or
			(len(input_feature_channel) != len(final_conv_channel) +1)):
			raise ValueError


		# 首先记录初始化的各个控制参数
		self.input_channel = output_channel
		self.input_feature_channel = input_feature_channel
		self.skip_connection = skip_connection
		self.middle_conv_channel = middle_conv_channel
		self.final_conv_channel = final_conv_channel



		# 初始化网络层（带参数层）---------------------------------------------------------
		self.backbone = backbone


		conv_modules_list = []
		# 首先添加最下层（因为输入为input的2个，因此单独添加
		conv_modules_list.append(
			self.make_conv_module(
				input_c=self.input_feature_channel[-1],
				skip_c=self.input_feature_channel[-2],
				middle_c=self.middle_conv_channel[-1],
				final_c=self.final_conv_channel[-1],
				skip_connection_flag=self.skip_connection[-1]
			))

		for index in range(len(self.skip_connection) - 1):
			conv_modules_list.append(
				self.make_conv_module(
					input_c=self.final_conv_channel[-(index + 1)],
					skip_c=self.input_feature_channel[-(index + 3)],
					middle_c=self.middle_conv_channel[-(index + 2)],
					final_c=self.final_conv_channel[-(index + 2)],
					skip_connection_flag=self.skip_connection[-(index + 2)]
				))


		self.conv_modules_list = nn.ModuleList(conv_modules_list) # 顺序是，从深层特征 conv到浅层的，即尺寸从小到大


		# 最后分割的head
		self.seg_head = nn.Conv3d(self.final_conv_channel[0], output_channel, kernel_size=1)

		# 初始化网络层（非参数层，即上采样层）--------------------------------------------------------
		# 上采样块数量n=len(input_feature_channel)
		self.upsample_list = nn.ModuleList([nn.Upsample(size=(2, 2, 2), mode='trilinear', align_corners=True)  # 这里3D的不太一样
											for _ in range(len(input_feature_channel))])

	def make_conv_module(self, input_c, skip_c, middle_c, final_c, skip_connection_flag):
		if skip_connection_flag:
			input_c = input_c + skip_c
		return VGGBlock3D(input_c, middle_c, final_c)

	def update_upsample_size(self, x):
		# x 是[ input, feature1, feature2, ...]这种形式
		for index in range(len(x)-1):
			self.upsample_list[index] = nn.Upsample(size=(x[-(index+2)].shape[2], x[-(index+2)].shape[3], x[-(index+2)].shape[4]), mode='trilinear', align_corners=True)
			# 顺序是从深层特征到浅层(倒数第二个特征一直到input），因为要和conv list保持一致

	def forward(self, x):
		# 将输入储存到list
		input = [x]

		# 获取层级特征
		backboneF = self.backbone(x)

		# 将输入和层级特征的两个list合并
		x = input + backboneF  # 即[ input, feature1, feature2, ...]这种形式

		# 这里一步到位，一次性调整所有上采样模块
		self.update_upsample_size(x)

		# 开始传火，首先对最下层特征操作
		reture_f_list = []
		if self.skip_connection[-1]:
			up_f = self.conv_modules_list[0](
				torch.cat([x[-2], self.upsample_list[0](x[-1])], axis=1)
			)
		else:
			up_f = self.conv_modules_list[0](
				self.upsample_list[0](x[-1])
			)
		reture_f_list.append(up_f)


		# 之后对其余层特征操作
		for index in range(len(self.skip_connection)-1):
			if self.skip_connection[-(index+2)]:
				up_f = self.conv_modules_list[index+1](
					torch.cat([x[-(index+3)], self.upsample_list[index+1](up_f)], axis=1)
				)
			else:
				up_f = self.conv_modules_list[index+1](
					self.upsample_list[index + 1](up_f)
				)
			reture_f_list.append(up_f)

		# 最后计算得到输出map
		final_out = self.seg_head(self.upsample_list[-1](up_f))


		# 返回各个尺寸的特征图（供debug用）
		return dict(segout = final_out, f_list = reture_f_list)

class nested_Unet_beta(nn.Module):
	pass

class nested_Unet_beta3D(nn.Module):
	pass



# reference to https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
# 很简洁的实现

class VGGBlock(nn.Module):
	"""
	简洁的CBA结构（Conv，bn，activation）
	输入输出形状保持一致
	"""
	def __init__(self, in_channels, middle_channels, out_channels, stride=1):
		super().__init__()
		self.in_channels  =in_channels
		self.out_channels  =out_channels

		# a
		self.relu = nn.ReLU(inplace=True)

		# cb1
		self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1, stride=stride)
		if norm_set == 'bn':
			self.bn1 = nn.BatchNorm2d(middle_channels)
		elif norm_set == 'gn':
			self.bn1 = nn.GroupNorm(gn_c, middle_channels)
		elif norm_set == 'in':
			self.bn1 = nn.InstanceNorm2d(middle_channels)

		# cb2
		self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1, stride=stride)
		if norm_set == 'bn':
			self.bn2 = nn.BatchNorm2d(out_channels)
		elif norm_set == 'gn':
			self.bn2 = nn.GroupNorm(gn_c, out_channels)
		elif norm_set == 'in':
			self.bn2 = nn.InstanceNorm2d(out_channels)


	def forward(self, x):

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		return out

class VGGBlock3D(nn.Module):
	"""
	简洁的CBA结构（Conv，bn，activation）
	输入输出形状保持一致
	"""
	def __init__(self, in_channels, middle_channels, out_channels, stride=1):
		super().__init__()
		self.in_channels  =in_channels
		self.out_channels  =out_channels



		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1, stride=stride)
		if norm_set == 'bn':
			self.bn1 = nn.BatchNorm3d(middle_channels)
		elif norm_set == 'gn':
			self.bn1 = nn.GroupNorm(gn_c, middle_channels)
		elif norm_set == 'in':
			self.bn1 = nn.InstanceNorm3d(middle_channels)




		self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1, stride=stride)
		self.bn2 = nn.BatchNorm3d(out_channels)
		if norm_set == 'bn':
			self.bn2 = nn.BatchNorm2d(out_channels)
		elif norm_set == 'gn':
			self.bn2 = nn.GroupNorm(gn_c, out_channels)
		elif norm_set == 'in':
			self.bn2 = nn.InstanceNorm2d(out_channels)


	def forward(self, x):

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		return out

class UNet(nn.Module):
	"""
	简洁的Unet结构，下采样使用pooling，上采样使用线性插值
	通道数设计也很简单
	"""
	def __init__(self, num_classes, input_channels=3, **kwargs):
		super().__init__()

		if 'nb_filter' in kwargs.keys():
			nb_filter = kwargs['nb_filter']
		else:
			nb_filter = [32, 64, 128, 256, 512]#

		self.pool = nn.MaxPool2d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
		self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

		self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
		self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
		self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

		self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


	def forward(self, input):

		x0_0 = self.conv0_0(input)
		x1_0 = self.conv1_0(self.pool(x0_0))
		x2_0 = self.conv2_0(self.pool(x1_0))
		x3_0 = self.conv3_0(self.pool(x2_0))

		x4_0 = self.conv4_0(self.pool(x3_0))


		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
		x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
		x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

		output = self.final(x0_4)
		return output

class UNet3D(nn.Module):
	"""
	简洁的Unet结构，下采样使用pooling，上采样使用线性插值
	通道数设计也很简单
	"""
	def __init__(self, num_classes, input_channels=3, **kwargs):
		super().__init__()

		if 'nb_filter' in kwargs.keys():
			nb_filter = kwargs['nb_filter']
		else:
			nb_filter = [32, 64, 128, 256, 512]#

		self.pool = nn.MaxPool3d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

		self.conv0_0 = VGGBlock3D(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = VGGBlock3D(nb_filter[0], nb_filter[1], nb_filter[1])
		self.conv2_0 = VGGBlock3D(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = VGGBlock3D(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = VGGBlock3D(nb_filter[3], nb_filter[4], nb_filter[4])

		self.conv3_1 = VGGBlock3D(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
		self.conv2_2 = VGGBlock3D(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
		self.conv1_3 = VGGBlock3D(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv0_4 = VGGBlock3D(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

		self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)


	def forward(self, input):

		x0_0 = self.conv0_0(input)
		x1_0 = self.conv1_0(self.pool(x0_0))
		x2_0 = self.conv2_0(self.pool(x1_0))
		x3_0 = self.conv3_0(self.pool(x2_0))

		x4_0 = self.conv4_0(self.pool(x3_0))


		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
		x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
		x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

		output = self.final(x0_4)
		return output

class NestedUNet(nn.Module):

	def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
		super().__init__()

		if 'nb_filter' in kwargs.keys():
			nb_filter = kwargs['nb_filter']
		else:
			nb_filter = [32, 64, 128, 256, 512]#


		self.deep_supervision = deep_supervision

		self.pool = nn.MaxPool2d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
		self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

		self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
		self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

		self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

		self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

		self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

		if self.deep_supervision:
			self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
			self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
		else:
			self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


	def forward(self, input):
		x0_0 = self.conv0_0(input)


		x1_0 = self.conv1_0(self.pool(x0_0))
		x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

		x2_0 = self.conv2_0(self.pool(x1_0))
		x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

		x3_0 = self.conv3_0(self.pool(x2_0))
		x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

		x4_0 = self.conv4_0(self.pool(x3_0))
		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

		if self.deep_supervision:
			output1 = self.final1(x0_1)
			output2 = self.final2(x0_2)
			output3 = self.final3(x0_3)
			output4 = self.final4(x0_4)
			return [output1, output2, output3, output4]

		else:
			output = self.final(x0_4)
			return output

class NestedUNet3D(nn.Module): #

	def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):#
		super().__init__()

		if 'nb_filter' in kwargs.keys():
			nb_filter = kwargs['nb_filter']
		else:
			nb_filter = [32, 64, 128, 256, 512]#

		self.deep_supervision = deep_supervision#

		self.pool = nn.MaxPool3d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

		self.conv0_0 = VGGBlock3D(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = VGGBlock3D(nb_filter[0], nb_filter[1], nb_filter[1])
		self.conv2_0 = VGGBlock3D(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = VGGBlock3D(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = VGGBlock3D(nb_filter[3], nb_filter[4], nb_filter[4])

		self.conv0_1 = VGGBlock3D(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_1 = VGGBlock3D(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_1 = VGGBlock3D(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
		self.conv3_1 = VGGBlock3D(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

		self.conv0_2 = VGGBlock3D(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_2 = VGGBlock3D(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_2 = VGGBlock3D(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

		self.conv0_3 = VGGBlock3D(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_3 = VGGBlock3D(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

		self.conv0_4 = VGGBlock3D(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

		if self.deep_supervision:
			self.final1 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
			self.final2 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
			self.final3 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
			self.final4 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
		else:
			self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)


	def forward(self, input):
		x0_0 = self.conv0_0(input)


		x1_0 = self.conv1_0(self.pool(x0_0))
		x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

		x2_0 = self.conv2_0(self.pool(x1_0))
		x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

		x3_0 = self.conv3_0(self.pool(x2_0))
		x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

		x4_0 = self.conv4_0(self.pool(x3_0))
		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

		if self.deep_supervision:
			output1 = self.final1(x0_1)
			output2 = self.final2(x0_2)
			output3 = self.final3(x0_3)
			output4 = self.final4(x0_4)
			return [output1, output2, output3, output4]

		else:
			output = self.final(x0_4)
			return output




if __name__ == '__main__':
	# 测试unet beta
	# back_bone = fake_backbone()
	# unet = Unet_beta(
	# 	output_channel=3,
	# 	backbone=back_bone,
	# 	input_feature_channel=[64, 128, 256, 512, 1024],  # 最后一个特征图开始上采样
	# 	skip_connection=[False, False, False, True],
	# 	middle_conv_channel=[64, 128, 256, 512],
	# 	final_conv_channel=[10, 20, 30, 40],
	# )
	# out = unet(torch.ones([3, 3, 512, 512]))

	# 测试unet beta3D
	# back_bone = fake_backbone3D()
	# unet = Unet_beta3D(
	# 	output_channel=3,
	# 	backbone=back_bone,
	# 	input_feature_channel=[2, 3, 4, 5, 6],  # 最后一个特征图开始上采样
	# 	skip_connection=[False, False, False, True],
	# 	middle_conv_channel=[32, 32, 32, 32],
	# 	final_conv_channel=[32, 32, 32, 32],
	# )
	# out = unet(torch.ones([3, 3, 2, 2,2]))

	# 测试unet
	# unet = UNet(num_classes=2, input_channels=1)
	# out = unet(torch.ones([1, 1, 256, 256]))


	# 测试unet3D
	# unet = UNet3D(num_classes=2, input_channels=1, nb_filter = [16,16,16,16,16])
	# with torch.no_grad():
	# 	out = unet(torch.ones([1, 1, 128, 128, 64]))


	# 测试unet++
	# unet = NestedUNet(num_classes=2, input_channels=1)
	# out = unet(torch.ones([1, 1, 256, 256]))


	# 测试unet++3D
	unet = NestedUNet3D(num_classes=2, input_channels=3, nb_filter = [16,16,16,16,16])
	with torch.no_grad():
		out = unet(torch.ones([1, 3, 96, 96, 48]))






