config = {
	'dataset': 'digits',
	'task_map': {k: k for k in range(10)}
}

config_init = {
	'max_samples': 18000
}

config_train_joint = {
	'batch_size': 128,
	'num_epochs': 20
}

config_train_seq = {
	'batch_size': 128,
	'num_epochs': 6
}