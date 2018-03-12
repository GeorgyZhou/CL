import numpy as np

disp_avlbl = True
from os import environ
if 'DISPLAY' not in environ:
	disp_avlbl = False
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(123)
import seaborn

########################### Color array ############################

# Array of 269 distinctive colors
dist_color_array = [
"#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
"#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
"#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
"#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
"#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",

"#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
"#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
"#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
"#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
"#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
"#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
"#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
"#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

"#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
"#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
"#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
"#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
"#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
"#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
"#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
"#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",

"#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
"#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"
]

########################### Helper functions ############################

def t_label_formatter(t_label):
	return t_label + 1

def accuracy_curves(eval_arr, task_map, unique_task_ids, savefile=None, fontsize=None):
	''' Plots accuracy curves for all tasks for one algorithm
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	# Compute number of tasks and unique tasks
	num_unique_tasks, num_tasks = eval_arr.shape

	# Plot forgetting curves
	plt.figure()
	handles = []
	for u_task in range(num_unique_tasks):
		h, = plt.plot(eval_arr[u_task,:], marker='o', 
				label='Task {}'.format(t_label_formatter(unique_task_ids[u_task])))
		handles.append(h)
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Task')
	plt.ylabel('Accuracy')
	# plt.legend(handles=handles, loc='center left')
	plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	plt.ylim((-0.01, 1.01))

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()

def accuracy_curves_many_tasks(eval_arr, task_map, unique_task_ids, savefile=None, fontsize=None):
	''' Plots accuracy curves for one algorithm when learning on 
		many tasks (max 269 unique colors supported)
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	# Compute number of tasks and unique tasks
	num_unique_tasks, num_tasks = eval_arr.shape

	# Plot forgetting curves
	plt.figure()
	handles = []
	for u_task in range(num_unique_tasks):
		h, = plt.plot(eval_arr[u_task,:], color=dist_color_array[u_task],
				label='Task {}'.format(t_label_formatter(unique_task_ids[u_task])))
		handles.append(h)
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Task')
	plt.ylabel('Accuracy')
	# plt.legend(handles=handles, loc='center left')
	plt.legend(handles=handles, bbox_to_anchor=(1.05, 1.0), loc=2, 
		ncol=int(np.ceil(num_unique_tasks/20.0)), borderaxespad=0.0)
	plt.ylim((-0.01, 1.01))

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def avg_accuracy_curve(eval_arr, task_map, unique_task_ids, savefile=None, fontsize=None):
	''' Plots accuracy curve averaged over all tasks for one algorithm
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	# Compute number of tasks and unique tasks
	num_unique_tasks, num_tasks = eval_arr.shape

	# Plot forgetting curve
	plt.figure()
	avg_eval_arr = np.mean(eval_arr, axis=0)
	plt.plot(avg_eval_arr, marker='o')
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Task')
	plt.ylabel('Avg task accuracy')
	plt.ylim((-0.01, 1.01))

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def avg_accuracy_curves(eval_arr_list, task_map, unique_task_ids, legend_list=None, savefile=None, fontsize=None):
	''' Plots average accuracy curves for multiple algorithms
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	plt.figure()
	i = 0
	for eval_arr in eval_arr_list:
		# Compute number of tasks and unique tasks
		num_unique_tasks, num_tasks = eval_arr.shape

		# Plot forgetting curve
		avg_eval_arr = np.mean(eval_arr, axis=0)
		if (legend_list is not None):
			plt.plot(avg_eval_arr, marker='o', label=legend_list[i])
			i += 1
		else:
			plt.plot(avg_eval_arr, marker='o')
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Tasks')
	plt.ylabel('Avg task accuracy')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	plt.ylim((-0.01, 1.01))

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def avg_forgetting_curves(eval_arr_list, task_map, unique_task_ids, legend_list=None, savefile=None, fontsize=None):
	''' Plots avg forgetting curve for multiple algorithms
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	plt.figure()
	i = 0
	for eval_arr in eval_arr_list:
		# Compute number of tasks and unique tasks
		num_unique_tasks, num_tasks = eval_arr.shape

		task_done_yet = []
		avg_eval_arr = np.zeros(num_tasks)

		for k in range(num_tasks):
			if (task_map[k] not in task_done_yet):
				task_done_yet.append(task_map[k])
			avg_eval_arr[k] = np.mean(eval_arr[task_done_yet, k])

		if (legend_list is not None):
			plt.plot(avg_eval_arr, marker='o', label=legend_list[i])
			i += 1
		else:
			plt.plot(avg_eval_arr, marker='o')
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Tasks')
	plt.ylabel('Avg task accuracy')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	plt.ylim((-0.01, 1.01))

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def avg_forgetting_curves_K(eval_arr_list, task_map, unique_task_ids, K, legend_list=None, savefile=None, fontsize=None):
	''' Plots avg forgetting curve for multiple algorithms on the last K tasks encountered.
		Assumes all task IDs are unique.
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	assert len({val for val in task_map.values()}) == len(unique_task_ids)
	assert K <= len(unique_task_ids)

	plt.figure()
	i = 0
	for eval_arr in eval_arr_list:
		# Compute number of tasks and unique tasks
		num_unique_tasks, num_tasks = eval_arr.shape
		assert num_tasks == num_unique_tasks

		avg_eval_arr = np.zeros(num_tasks)

		for k in range(num_tasks):
			if k >= K:
				avg_eval_arr[k] = np.mean(eval_arr[k-K+1:k+1, k])
			else:
				avg_eval_arr[k] = np.mean(eval_arr[0:k+1, k])

		if (legend_list is not None):
			plt.plot(avg_eval_arr, marker='o', label=legend_list[i])
			i += 1
		else:
			plt.plot(avg_eval_arr, marker='o')
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Tasks')
	plt.ylabel('Avg task accuracy on last {} tasks'.format(K))
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	plt.ylim((-0.01, 1.01))

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def time_curve(tr_time, task_map, accumulate=True, savefile=None, fontsize=None):
	''' Plots time vs tasks for one algorithm
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	# Compute number of tasks and unique tasks
	num_tasks = tr_time.shape[0]

	# Add up time if needed
	if accumulate:
		time_array = np.cumsum(tr_time)
	else:
		time_array = tr_time

	# Plot forgetting curve
	plt.figure()
	plt.plot(time_array)
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Task')
	plt.ylabel('Time taken')

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def time_curves(tr_time_list, task_map, accumulate=True, legend_list=None, savefile=None, fontsize=None):
	''' Plots time vs tasks for multiple algorithms
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	plt.figure()
	i = 0
	for tr_time in tr_time_list:
		# Compute number of tasks and unique tasks
		num_tasks = tr_time.shape[0]

		# Add up time if needed
		if accumulate:
			time_array = np.cumsum(tr_time)
		else:
			time_array = tr_time
		if (legend_list is not None):
			plt.plot(time_array, marker='o', label=legend_list[i])
			i += 1
		else:
			plt.plot(time_array, marker='o')

	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Task')
	plt.ylabel('Time taken (seconds)')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def resilience_curve(eval_arr, factor_vals, labels, savefile=None, fontsize=None):
	''' Plots resilience curves for noise/ambience/occlusion
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	# Plot resilience curve
	plt.figure()
	plt.plot(factor_vals, eval_arr)
	plt.xlabel(labels['x'])
	plt.ylabel(labels['y'])

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()


def resilience_curves(eval_arr_list, factor_vals, labels, legend_list, savefile=None, fontsize=None):
	''' Plots resilience curves for noise/ambience/occlusion
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	plt.figure()
	i = 0

	for eval_arr in eval_arr_list:
		if (legend_list is not None):
			plt.plot(factor_vals, eval_arr, marker='o', label=legend_list[i])
			i += 1
		else:
			plt.plot(factor_vals, eval_arr, marker='o')

	plt.xlabel(labels['x'])
	plt.ylabel(labels['y'])
	plt.legend()

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()

def repetition_curves(eval_arr, task_map, unique_task_ids, savefile=None, fontsize=None):
	''' Plots repetition curves for all tasks for one algorithm
	'''
	# Set font size
	if fontsize is not None:
		default_fontsize = plt.rcParams.get('font.size')
		plt.rcParams.update({'font.size': fontsize})

	# Compute number of tasks and unique tasks
	num_unique_tasks, num_tasks = eval_arr.shape

	# Compute repeated tasks
	unique_tasks, task_counts = np.unique([val for val in task_map.values()], return_counts=True)
	repeat_tasks_list = unique_tasks[task_counts > 1]

	# Plot forgetting curves
	plt.figure()
	handles = []
	for r_task in range(len(repeat_tasks_list)):
		r_task_id = repeat_tasks_list[r_task]
		h, = plt.plot(eval_arr[r_task_id,:], marker='o', 
				label='Task {}'.format(t_label_formatter(r_task_id)))
		handles.append(h)
	xticks = ['Task {}'.format(t_label_formatter(val)) for val in task_map.values()]
	plt.xticks(range(num_tasks), xticks, rotation='vertical')
	plt.xlabel('Task')
	plt.ylabel('Accuracy')
	# plt.legend(handles=handles, loc='center left')
	plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	plt.ylim((-0.01, 1.01))

	# Save/display figure
	if savefile is None:
		plt.show()
	else:
		plt.savefig(savefile, bbox_inches='tight')

	# Reset font size
	if fontsize is not None:
		plt.rcParams.update({'font.size': default_fontsize})
	plt.close()