# 相对于decoder的，例如fake_backbone这些



import torch.nn as nn
import torch
import torch.nn.functional as F
from decoder import VGGBlock



# 简单网络，适合分辨率小的数据（32*32，或28*28）
class custom_VGG2D(nn.Module):
	"""
	简洁的vgg结构
	"""
	def __init__(self, input_channels=3, nb_filter = [16,32,64], downsample_ration = [0.8, 0.8, 0.8], conv_first = True):
		"""
		深度由nb_filter的len决定，list长度为3，则网络包含3层
		逻辑（mode）有两种：conv_first 先卷积再下采样，否则就先下采样再卷积
		:param input_channels:
		:param nb_filter:
		:param downsample_ration: 每个元素最好小于等于1，否则会变成上采样
		"""

		super().__init__()
		self.nb_filter = nb_filter
		self.downsample_ration = downsample_ration
		self.conv_first = conv_first

		tmp_c = [input_channels] + nb_filter
		self.conv_list = nn.ModuleList([
			VGGBlock(tmp_c[index], tmp_c[index+1], tmp_c[index+1])
			for index in range(len(self.nb_filter))
		])

	def forward(self, x):
		reture_f_list = []
		for i in range(len(self.nb_filter)):
			if self.conv_first:
				x = F.interpolate(self.conv_list[i](x),
								  scale_factor=self.downsample_ration[i],
								  mode='bicubic',
								  align_corners=True)
			else:
				x = self.conv_list[i](
					F.interpolate(x, scale_factor=self.downsample_ration[i],
								  mode='bicubic',
								  align_corners=True))

			reture_f_list.append(x)
		return reture_f_list


# 3D 简单网络



# 3D 复杂网络（3D resnest）




if __name__ == '__main__':
	input = torch.ones([3,3,32,32])
	# net = custom_VGG2D(3, [32,64,128], [0.8, 0.8, 0.8], True)
	net = custom_VGG2D(3, [32,64,128], [0.8, 0.8, 0.8], False)
	resize_input = net(input)
	_ = [print(i.shape) for i in resize_input]












