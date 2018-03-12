from __future__ import print_function
import os
import errno
import importlib.util
import argparse
import pdb
import time
import pickle
import shutil

import sys
sys.path.append('./')
from utils.plot_utils import *
from utils.transfer_utils import *

import numpy as np
np.random.seed(123)

############################## Main method ###############################

def main(args):
	# Create output directory if it does not exist
	try:
		os.makedirs(args.outdir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	# Decide training mode
	if args.sequential:
		trainmode = 'seq'
	elif args.batchseq:
		trainmode = 'batchseq'
	elif args.joint:
		trainmode = 'joint'
	else:
		trainmode = 'fulljoint'

	# Import config
	spec = importlib.util.spec_from_file_location('config', args.configfile)
	conf_mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(conf_mod)
	config = conf_mod.config
	config_init = conf_mod.config_init
	config_train = conf_mod.config_train_joint if (trainmode != 'seq' and trainmode != 'batchseq') else conf_mod.config_train_seq

	# Import model
	spec = importlib.util.spec_from_file_location('model', args.modelfile)
	model_mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(model_mod)

	# Save configs, game, models, game state and numpy state
	np_rnd_state = np.random.get_state()
	pickle.dump(config, file=open(args.outdir+'config.pkl', 'wb'))
	pickle.dump(config_init, file=open(args.outdir+'config_init.pkl', 'wb'))
	pickle.dump(config_train, file=open(args.outdir+'config_train.pkl', 'wb'))
	pickle.dump(np_rnd_state, file=open(args.outdir+'np_rnd_state.pkl', 'wb'))
	shutil.copy(src=args.configfile, dst=args.outdir+'configfile.py')
	shutil.copy(src=args.modelfile, dst=args.outdir+'modelfile.py')

	# Generate model
	model = model_mod.CLModel(**config_init)

	# Compute number of tasks and unique tasks
	num_tasks = len(config['task_map'])
	unique_task_ids = list(set(val for val in config['task_map'].values()))
	num_unique_tasks = len(unique_task_ids)

	# Load all task data
	all_tasks_data = {}
	for task in range(num_unique_tasks):
		if unique_task_ids[task] not in all_tasks_data:			
			task_data = np.load(args.datadir + config['dataset'] + 
						'_{}.npz'.format(unique_task_ids[task]))
			(x_train, y_train, x_test, y_test) = (task_data['x_train'], 
				task_data['y_train'], task_data['x_test'], task_data['y_test'])
			n_train = x_train.shape[0]
			n_test = x_test.shape[0]
			# Shuffle data
			tr_pos = np.random.permutation(x_train.shape[0])
			te_pos = np.random.permutation(x_test.shape[0])
			x_train = x_train[tr_pos]
			y_train = y_train[tr_pos]
			x_test = x_test[te_pos]
			y_test = y_test[te_pos]
			# Store data
			all_tasks_data[unique_task_ids[task]] = (x_train, y_train, x_test, y_test)

	# Result arrays
	task_eval_tr = np.zeros((num_unique_tasks, num_tasks), dtype='float32')
	task_eval_te = np.zeros((num_unique_tasks, num_tasks), dtype='float32')
	FWT_tr = 0.0
	BWT_tr = 0.0
	FWT_te = 0.0
	BWT_te = 0.0
	ACC_tr = 0.0
	ACC_te = 0.0
	tr_time = np.zeros(num_tasks, dtype='float32')
	tr_process_time = np.zeros(num_tasks, dtype='float32')

	# Temporary variables required for several trainmodes
	if trainmode == 'joint' or trainmode == 'fulljoint':
		joint_task_data = []
		joint_task_dict = {}
	elif trainmode == 'batchseq':
		train_freq = config_init['LTM_train_freq']
		train_iter = 0
		batch_task_data = []
		batch_task_dict = {}

	# Train on tasks while evaluating side-by-side
	for task in range(num_tasks):
		t_id = config['task_map'][task]
		
		# Trainmode: 'seq'
		if trainmode == 'seq':
			tic = time.time()
			tic_process = time.process_time()
			print('Training on Task {} [t_id={}]:'.format(task, t_id))
			history = model.fit(x=all_tasks_data[t_id][0], y=all_tasks_data[t_id][1],
				batch_size=config_train['batch_size'], epochs=config_train['num_epochs'], 
				verbose=args.verbose, traintype='seq', task_dict={t_id: 1})
			print('Training finished')
			toc_process = time.process_time()
			toc = time.time()
			tr_time[task] = toc - tic
			tr_process_time[task] = toc_process - tic_process

		# Trainmode: 'batchseq'
		elif trainmode == 'batchseq':
			if train_iter % train_freq == 0:
				batch_task_data = [all_tasks_data[t_id][0], all_tasks_data[t_id][1],
							all_tasks_data[t_id][2], all_tasks_data[t_id][3]]
				batch_task_dict = {t_id: 1}
			else:
				batch_task_data = [np.concatenate((batch_task_data[0], all_tasks_data[t_id][0]), axis=0),
							np.concatenate((batch_task_data[1], all_tasks_data[t_id][1]), axis=0),
							np.concatenate((batch_task_data[2], all_tasks_data[t_id][2]), axis=0),
							np.concatenate((batch_task_data[3], all_tasks_data[t_id][3]), axis=0)]
				if t_id in batch_task_dict:
					batch_task_dict[t_id] += 1
				else:
					batch_task_dict[t_id] = 1
			train_iter = (train_iter + 1) % train_freq

			if train_iter % train_freq == 0:
				tic = time.time()
				tic_process = time.process_time()
				history = model.fit(x=batch_task_data[0], y=batch_task_data[1],
					batch_size=config_train['batch_size'], epochs=config_train['num_epochs'], 
					verbose=args.verbose, traintype='seq', task_dict=batch_task_dict)
				toc_process = time.process_time()
				toc = time.time()
				tr_time[task] = toc - tic
				tr_process_time[task] = toc_process - tic_process

		# Trainmode: 'joint' (does data processing for 'fulljoint')
		elif trainmode == 'joint' or trainmode == 'fulljoint':
			if task == 0:
				joint_task_data = [all_tasks_data[t_id][0],
									all_tasks_data[t_id][1],
									all_tasks_data[t_id][2],
									all_tasks_data[t_id][3]]
				joint_task_dict = {t_id: 1}
			else:
				joint_task_data = [np.concatenate((joint_task_data[0], all_tasks_data[t_id][0]), axis=0),
									np.concatenate((joint_task_data[1], all_tasks_data[t_id][1]), axis=0),
									np.concatenate((joint_task_data[2], all_tasks_data[t_id][2]), axis=0),
									np.concatenate((joint_task_data[3], all_tasks_data[t_id][3]), axis=0)]
				if t_id in joint_task_dict:
					joint_task_dict[t_id] += 1
				else:
					joint_task_dict[t_id] = 1
			if trainmode == 'joint':
				model = model_mod.CLModel(**config_init)
				tic = time.time()
				tic_process = time.process_time()
				history = model.fit(x=joint_task_data[0], y=joint_task_data[1], 
					batch_size=config_train['batch_size'], epochs=config_train['num_epochs'], 
					verbose=args.verbose, traintype='joint', task_dict=joint_task_dict)
				toc_process = time.process_time()
				toc = time.time()
				tr_time[task] = toc - tic
				tr_process_time[task] = toc_process - tic_process
		
		# Evaluate model on all unique tasks
		if trainmode == 'seq' or trainmode == 'batchseq' or trainmode == 'joint':
			for eval_task in range(num_unique_tasks):
				eval_t_id = unique_task_ids[eval_task]
				print('Evaluating on task {}:'.format(eval_t_id))
				task_eval_tr[eval_task, task] = model.evaluate(
						x=all_tasks_data[eval_t_id][0], y=all_tasks_data[eval_t_id][1], 
						batch_size=config_train['batch_size'], verbose=0, t_id=eval_t_id)[1]
				task_eval_te[eval_task, task] = model.evaluate(
						x=all_tasks_data[eval_t_id][2], y=all_tasks_data[eval_t_id][3], 
						batch_size=config_train['batch_size'], verbose=0, t_id=eval_t_id)[1]

		# Print some info
		if trainmode == 'seq' or trainmode == 'batchseq' or trainmode == 'joint':
			print('Task {} [t_id {}]: Total time = {}s, Process time = {}s'.format(task, t_id, tr_time[task], tr_process_time[task]))
			print('Accuracy on all training sets: {}'.format(task_eval_tr[:, task]))
			print('Accuracy on all test sets: {}'.format(task_eval_te[:, task]))			

		# Visualize model
		if (trainmode == 'seq' or trainmode == 'batchseq' or trainmode == 'joint') and args.visualize:
			model.visualize(n=100, boxsize=10, savefile=args.outdir+trainmode+'_viz_step{}.png'.format(task))

	# Final evaluation
	if trainmode == 'seq' or trainmode == 'batchseq' or trainmode == 'joint':
		if num_tasks == num_unique_tasks:
			FWT_tr = FwdTransfer(task_eval_tr, baselines=None)
			FWT_te = FwdTransfer(task_eval_te, baselines=None)
			BWT_tr = BwdTransfer(task_eval_tr)
			BWT_te = BwdTransfer(task_eval_te)
		ACC_tr = Accuracy(task_eval_tr)
		ACC_te = Accuracy(task_eval_te)
		print('ACC_tr: {}, ACC_te: {}'.format(ACC_tr, ACC_te))
		if num_tasks == num_unique_tasks:
			print('FWT_tr: {}, FWT_te: {}, BWT_tr: {}, BWT_te: {}'.format(FWT_tr, FWT_te, BWT_tr, BWT_te))

	# For fulljoint trainmode
	if trainmode == 'fulljoint':
		# Train model
		tic = time.time()
		tic_process = time.process_time()
		history = model.fit(x=joint_task_data[0], y=joint_task_data[1], 
				batch_size=config_train['batch_size'], epochs=config_train['num_epochs'], 
				verbose=args.verbose, traintype='joint', task_dict=joint_task_dict)
		toc_process = time.process_time()
		toc = time.time()
		tr_time = toc - tic
		tr_process_time = toc_process - tic_process

		# Evaluate
		train_score = model.evaluate(x=joint_task_data[0], y=joint_task_data[1], 
				batch_size=config_train['batch_size'], verbose=0)
		test_score = model.evaluate(x=joint_task_data[2], y=joint_task_data[3], 
				batch_size=config_train['batch_size'], verbose=0)

		# Display results
		print('Full joint training: Total time = {}s, Process time = {}s'.format(tr_time, tr_process_time))
		print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
		print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

		# Visualize model
		if args.visualize:
			model.visualize(n=100, boxsize=10, savefile=args.outdir+trainmode+'_viz.png')

				
	# Save results
	if trainmode == 'seq' or trainmode == 'batchseq' or trainmode == 'joint':
		kwds = {'trainmode': trainmode,
				'task_eval_tr': task_eval_tr,
				'task_eval_te': task_eval_te,
				'task_map': config['task_map'],
				'unique_task_ids': unique_task_ids,
				'FWT_tr': FWT_tr,
				'FWT_te': FWT_te,
				'BWT_tr': BWT_tr,
				'BWT_te': BWT_te,
				'ACC_tr': ACC_tr,
				'ACC_te': ACC_te,
				'tr_time': tr_time,
				'tr_process_time': tr_process_time}
	elif trainmode == 'fulljoint':
		kwds = {'trainmode': trainmode,
				'train_score': train_score,
				'test_score': test_score,
				'tr_time': tr_time,
				'tr_process_time': tr_process_time}
	np.savez_compressed(args.outdir+trainmode+'.npz', **kwds)
	model.save_weights(fileprefix=args.outdir+'weights_'+trainmode, overwrite=True)

	# Plot curves
	if trainmode == 'seq' or trainmode == 'batchseq' or trainmode == 'joint':
		# Forgetting curves
		accuracy_curves(task_eval_tr, config['task_map'], unique_task_ids, savefile=args.outdir+trainmode+'_forgetting_curves_tr.png')
		accuracy_curves(task_eval_te, config['task_map'], unique_task_ids, savefile=args.outdir+trainmode+'_forgetting_curves_te.png')
		# Avg forgetting curves
		avg_accuracy_curve(task_eval_tr, config['task_map'], unique_task_ids, savefile=args.outdir+trainmode+'_avg_forgetting_curve_tr.png')
		avg_accuracy_curve(task_eval_te, config['task_map'], unique_task_ids, savefile=args.outdir+trainmode+'_avg_forgetting_curve_te.png')
		# Timing curves
		time_curve(tr_time, config['task_map'], True, savefile=args.outdir+trainmode+'_time.png')
		time_curve(tr_process_time, config['task_map'], True, savefile=args.outdir+trainmode+'_process_time.png')

############################## Main code ###############################

if __name__ == '__main__':
	# Initial time
	t_init = time.time()

	parser = argparse.ArgumentParser(description=
		'Train model on dataset')
	parser.add_argument('-d', '--datadir', 
		help='Input directory for dataset')
	parser.add_argument('-m', '--modelfile', 
		help='Input file for model')
	parser.add_argument('-o', '--outdir', 
		help='Output directory for results')
	parser.add_argument('-c', '--configfile',
		help='Input file for parameters, constants and initial settings')
	parser.add_argument('-viz', '--visualize',
		help='Visualize (only if model is visualization capable)',
		action='store_true')
	parser.add_argument('-v', '--verbose',
		help='Increase output verbosity',
		action='store_true')
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-j", "--joint",
		help='Train jointly by passing in increasing number of tasks',
		action="store_true")
	group.add_argument("-s", "--sequential", 
		help='Train sequentially by passing in single tasks',
		action="store_true")
	group.add_argument("-f", "--fulljoint", 
		help='Train jointly on all tasks',
		action="store_true")
	group.add_argument("-b", "--batchseq", 
		help='Train sequentially on tasks grouped in batches',
		action="store_true")
	args = parser.parse_args()
	main(args)

	# Final time
	t_final = time.time()
	print('Progam finished in {} secs.'.format(t_final - t_init))