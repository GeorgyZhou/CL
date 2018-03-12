import numpy as np

disp_avlbl = True
from os import environ
if 'DISPLAY' not in environ:
	disp_avlbl = False
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn

########################### Transfer functions ############################

def FwdTransfer(task_eval_arr, baselines=None):
	''' Computes forward transfer statistic across all tasks for one algorithm

	Args:
		task_eval_arr: Array of task accuracies of shape (num_unique_tasks, num_tasks)
		baselines: Baseline accuracies on all tasks before any training

	Returns:
		FWT: Forward Transfer Statistic [GEM, NIPS'17]
	'''
	num_unique_tasks, num_tasks = task_eval_arr.shape
	assert num_unique_tasks == num_tasks # FWT defined only for unique tasks

	if baselines is None:
		baselines = np.ones((num_unique_tasks,)) / num_unique_tasks
	assert baselines.shape[0] == num_unique_tasks

	FWT = 0.0
	for task in range(1, num_unique_tasks):
		FWT += (task_eval_arr[task, task - 1] - baselines[task])
	return FWT / (num_unique_tasks - 1)

def BwdTransfer(task_eval_arr):
	''' Computes forward transfer statistic across all tasks for one algorithm

	Args:
		task_eval_arr: Array of task accuracies of shape (num_unique_tasks, num_tasks)

	Returns:
		BWT: Backward Transfer Statistic [GEM, NIPS'17]
	'''
	num_unique_tasks, num_tasks = task_eval_arr.shape
	assert num_unique_tasks == num_tasks # BWT defined only for unique tasks

	BWT = 0.0
	for task in range(0, num_unique_tasks-1):
		BWT += (task_eval_arr[task, -1] - task_eval_arr[task, task])
	return BWT / (num_unique_tasks - 1)

def Accuracy(task_eval_arr):
	''' Computes accuracy across all tasks for one algorithm

	Args:
		task_eval_arr: Array of task accuracies of shape (num_unique_tasks, num_tasks)

	Returns:
		ACC: Accuracy [GEM, NIPS'17]
	'''
	return np.mean(task_eval_arr[:, -1])