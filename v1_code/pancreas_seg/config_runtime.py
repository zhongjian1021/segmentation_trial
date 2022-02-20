config = dict(
	random_seed = 202020202,

	input_size = [96,96,48],
	batch_size = 6,
	batch_size_test = 8,


	# 学习率与优化器相关参数
	lr_min = 1e-12,
	lr_max = 1e-3,
	epoch_cosinedecay = 100,
	epoch_warmup = 5,
	adam_beta1 = 0.5,
	adam_beta2 = 0.999,

	# 其他参数
	epoch_num = 100,
	save_pth = r'/content/drive/MyDrive/v1_data/afte_process/',
	gpu_device_index = 0,
	aug_p = 0.6,


	val_flag = True,
	val_save_img = True,
	val_start_epoch = 1,
	val_step_epoch = 1,

	test_flag = True,
	test_save_img = True,
	test_start_epoch = 1,
	test_step_epoch = 1,

	model_save_step = 1,

	logger_print2pic_step_iter = 2,  # 将训练过程中的loss和测试验证指标保存为图片的时间步长



)