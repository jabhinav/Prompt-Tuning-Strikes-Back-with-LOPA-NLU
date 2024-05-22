import matplotlib.pyplot as plt
from plots.plot_data import *
import numpy as np

# Custom colors for the three performance plots
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
FONT_SIZE = 18  # Change 12 to the desired font size for A4 PDF
FIG_SIZE = (8, 6)  # Change 8, 6 to the desired figure size for A4 PDF


def baseline_perf_comparison():
	
	# Sample data for three plots
	for task in ['mrpc', 'sst2', 'rte', 'crux-o']:
		# Sample data for three plots
		soft_prompt_length = [5, 10, 25, 50, 100]
		if task == 'mrpc':
			perf_idpg = idpg_on_mrpc.values()
			perf_pt = pt_on_mrpc.values()
			perf_ours = ours_on_mrpc.values()
			perf_lora = lora_on_mrpc
		elif task == 'sst2':
			perf_idpg = idpg_on_sst2.values()
			perf_pt = pt_on_sst2.values()
			perf_ours = ours_on_sst2.values()
			perf_lora = lora_on_sst2
		elif task == 'rte':
			perf_idpg = idpg_on_rte.values()
			perf_pt = pt_on_rte.values()
			perf_ours = ours_on_rte.values()
			perf_lora = lora_on_rte
		elif task == 'crux-o':
			perf_idpg = idpg_on_cruxevalO.values()
			perf_pt = pt_on_cruxevalO.values()
			perf_ours = ours_r2_on_cruxevalO.values()  # We chose r=2 for CruxEval-O
			perf_lora = lora_on_cruxevalO
		else:
			raise ValueError(f"Invalid task: {task}")
		
		y_label = 'GLUE Performance (%)' if task in ['mrpc', 'sst2', 'rte'] else 'pass@1'
	
		# Set font size
		plt.rcParams.update({'font.size': FONT_SIZE})  # Change 12 to the desired font size for A4 PDF
		
		# Plotting the three plots
		plt.figure(figsize=FIG_SIZE)
		
		# Evenly spaced x-axis values
		x_values = np.arange(len(soft_prompt_length))
		
		# Plot IDPG
		plt.plot(x_values, perf_idpg, marker='o', label='S-IDPG', color=colors[0])
		# Plot PT
		plt.plot(x_values, perf_pt, marker='s', label='PT', color=colors[1])
		# Plot Ours
		plt.plot(x_values, perf_ours, marker='^', label='Ours', color=colors[2])
		
		# Plotting baseline
		plt.plot(x_values, [perf_lora] * len(soft_prompt_length), linestyle='--', color='gray', label='LoRA')
		
		# Adding x-axis labels
		plt.xticks(x_values, soft_prompt_length)
		
		# Adding labels
		plt.xlabel('Soft Prompt Length (t)')
		plt.ylabel(y_label, color='black')
		plt.title(f'{task.upper()}')
		plt.grid(True)
		
		# Displaying legend below the x-axis
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=4)
		
		# Adjust layout to prevent overlap of labels
		plt.tight_layout()
		
		# Saving the plot
		plt.savefig(f'plots/{task}.png', dpi=500)
		
		# Clear the plot
		plt.clf()


def ablation_on_rank_glue():
	ranks = [1, 2, 4]
	perf_mrpc = rank_on_mrpc.values()
	perf_rte = rank_on_rte.values()
	perf_sst2 = rank_on_sst2.values()
	
	# Set font size
	plt.rcParams.update({'font.size': FONT_SIZE})  # Change 12 to the desired font size for A4 PDF
	
	# Plotting the three plots
	plt.figure(figsize=FIG_SIZE)
	
	# Evenly spaced x-axis values
	x_values = np.arange(len(ranks))
	
	# Turn off decimals since rank is an integer
	plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
	
	plt.plot(x_values, perf_mrpc, marker='o', label='MRPC', color=colors[0])
	plt.plot(x_values, perf_rte, marker='s', label='RTE', color=colors[1])
	plt.plot(x_values, perf_sst2, marker='^', label='SST-2', color=colors[2])
	
	# Adding x-axis labels
	plt.xticks(x_values, ranks)
	
	# Adding labels
	plt.xlabel('Rank (r)')
	plt.ylabel('GLUE Performance (%)')
	# plt.title('Ablation on Rank')
	plt.grid(True)
	
	# Displaying legend
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
	
	# Adjust layout to prevent overlap of labels
	plt.tight_layout()
	
	# Saving the plot
	plt.savefig(f'plots/ablation_rank_on_nlu.png', dpi=500)


def baseline_perf_w_param_comparison():
	
	for task in ['mrpc', 'sst2', 'rte', 'crux-o']:
		# Sample data for three plots
		soft_prompt_length = [5, 10, 25, 50, 100]
		if task == 'mrpc':
			perf_idpg = idpg_on_mrpc.values()
			perf_pt = pt_on_mrpc.values()
			perf_ours = ours_on_mrpc.values()
			perf_lora = lora_on_mrpc
			
			# Sample data for number of parameters
			params_idpg = glue_idpg_params.values()
			params_ours = glue_ours_r4_params.values()
			params_pt = glue_pt_params.values()
		elif task == 'sst2':
			perf_idpg = idpg_on_sst2.values()
			perf_pt = pt_on_sst2.values()
			perf_ours = ours_on_sst2.values()
			perf_lora = lora_on_sst2
			
			# Sample data for number of parameters
			params_idpg = glue_idpg_params.values()
			params_ours = glue_ours_r4_params.values()
			params_pt = glue_pt_params.values()
		elif task == 'rte':
			perf_idpg = idpg_on_rte.values()
			perf_pt = pt_on_rte.values()
			perf_ours = ours_on_rte.values()
			perf_lora = lora_on_rte
			
			# Sample data for number of parameters
			params_idpg = glue_idpg_params.values()
			params_ours = glue_ours_r4_params.values()
			params_pt = glue_pt_params.values()
		elif task == 'crux-o':
			perf_idpg = idpg_on_cruxevalO.values()
			perf_pt = pt_on_cruxevalO.values()
			perf_ours = ours_r2_on_cruxevalO.values()  # We chose r=2 for CruxEval-O
			perf_lora = lora_on_cruxevalO
			
			# Sample data for number of parameters
			params_idpg = crux_idpg_params.values()
			params_ours = crux_ours_r2_params.values()
			params_pt = crux_pt_params.values()
		else:
			raise ValueError(f"Invalid task: {task}")
		
		y_label = 'GLUE Performance (%)' if task in ['mrpc', 'sst2', 'rte'] else 'pass@1'

		# Rescale the params such that largest is 100 and rest are scaled accordingly
		max_params = max(max(params_idpg), max(params_ours), max(params_pt))
		params_idpg = [100 * x / max_params for x in params_idpg]
		params_ours = [100 * x / max_params for x in params_ours]
		params_pt = [100 * x / max_params for x in params_pt]
		
		# Set font size
		plt.rcParams.update({'font.size': FONT_SIZE})  # Change 12 to the desired font size for A4 PDF
		
		# Plotting the performance plots
		fig, ax1 = plt.subplots(figsize=FIG_SIZE)
		
		# Evenly spaced x-axis values
		x_values = np.arange(len(soft_prompt_length))
		
		ax1.plot(x_values, perf_idpg, marker='o', label='S-IDPG', color=colors[0])
		ax1.plot(x_values, perf_pt, marker='s', label='PT', color=colors[1])
		ax1.plot(x_values, perf_ours, marker='^', label='Ours', color=colors[2])
		
		# Plotting baseline
		ax1.plot(x_values, [perf_lora] * len(soft_prompt_length), linestyle='--', color='gray', label='LoRA')
		
		ax1.set_xticks(x_values)
		ax1.set_xticklabels(soft_prompt_length)
		
		# Adding labels and legend for performance plots
		ax1.set_xlabel('Soft Prompt Length (m)')
		ax1.set_ylabel(y_label, color='black')
		ax1.tick_params(axis='y', labelcolor='black')
		# ax1.set_title('Performance vs Soft Prompt Length')
	
		# Displaying legend for performance plots below the plot
		# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
		
		# Plotting the bar plots for number of parameters
		ax2 = ax1.twinx()
		bar_width = 0.2
		
		# Adjusting the position of bars for better visibility
		bar_pos_1 = x_values - 2*bar_width
		bar_pos_2 = x_values - bar_width
		bar_pos_3 = x_values
		
		# Bar plots for number of parameters
		ax2.bar(bar_pos_1, params_pt, bar_width, label='PT Params', color=colors[1], alpha=0.4)
		ax2.bar(bar_pos_2, params_ours, bar_width, label='Ours Params', color=colors[2], alpha=0.4)
		ax2.bar(bar_pos_3, params_idpg, bar_width, label='S-IDPG Params', color=colors[0], alpha=0.4)
		
		# Adding labels and legend for number of parameters
		ax2.set_ylabel('Relative Number of Parameters (%)', color='black')
		ax2.tick_params(axis='y', labelcolor='black')
		
		# Displaying legend for number of parameters
		# ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=3)
		
		# Adding grid lines to both axes
		ax1.grid(True)
		ax2.grid(False)  # Don't show grid lines for number of parameters
		
		# Adjust layout to prevent overlap of labels
		plt.tight_layout()
		
		# Save the plot in high resolution
		plt.savefig(f'plots/{task}_w_param.png', dpi=500)  # Change the file format and dpi as needed
		
		# Clear the plot
		plt.clf()


	
	
def ablation_on_rank_glue_w_param():
	ranks = [1, 2, 4]
	perf_mrpc = rank_on_mrpc.values()
	perf_rte = rank_on_rte.values()
	perf_sst2 = rank_on_sst2.values()
	
	# Sample data for number of parameters
	params = glue_ours_t10_params.values()
	
	# Rescale the params such that largest is 100 and rest are scaled accordingly
	max_params = max(params)
	
	# Set font size
	plt.rcParams.update({'font.size': FONT_SIZE})  # Change 12 to the desired font size for A4 PDF
	
	# Plotting the performance plots
	fig, ax1 = plt.subplots(figsize=FIG_SIZE)
	
	# Evenly spaced x-axis values
	x_values = np.arange(len(ranks))
	
	ax1.plot(x_values, perf_mrpc, marker='o', label='MRPC', color=colors[0])
	ax1.plot(x_values, perf_rte, marker='s', label='RTE', color=colors[1])
	ax1.plot(x_values, perf_sst2, marker='^', label='SST-2', color=colors[2])
	
	ax1.set_xticks(x_values)
	ax1.set_xticklabels(ranks)
	
	# Adding labels and legend for performance plots
	ax1.set_xlabel('Rank (r)')
	ax1.set_ylabel('GLUE Performance (%)', color='black')
	ax1.tick_params(axis='y', labelcolor='black')
	# ax1.set_title('Performance vs Rank')
	
	# Plotting the bar plots for number of parameters
	ax2 = ax1.twinx()
	bar_width = 0.2
	
	# Adjusting the position of bars for
	bar_pos_1 = x_values
	
	# Bar plots for number of parameters
	ax2.bar(bar_pos_1, params, bar_width, label='Params', color=colors[4], alpha=0.2)
	
	# Adding labels and legend for number of parameters
	ax2.set_ylabel('Parameters (in millions)', color='black')
	ax2.tick_params(axis='y', labelcolor='black')
	
	# Combine legends and place below the plot
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	# ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
	
	# Adding grid lines to both axes
	ax1.grid(True)
	ax2.grid(False)  # Don't show grid lines for number of parameters
	
	# Adjust layout to prevent overlap of labels
	plt.tight_layout()
	
	# Save the plot in high resolution
	plt.savefig(f'plots/ablation_rank_on_nlu_w_param.png', dpi=500)  # Change the file format and dpi as needed
	

def ablation_on_rank_w_param_cruxeval():
	ranks = [1, 2, 4]
	perf_cruxI_10t = rank_on_cruxevalI_10t.values()
	perf_cruxO_10t = rank_on_cruxevalO_10t.values()
	perf_cruxI_50t = rank_on_cruxevalI_50t.values()
	perf_cruxO_50t = rank_on_cruxevalO_50t.values()
	
	# Sample data for number of parameters
	params_10t = crux_ours_t10_params.values()
	params_50t = crux_ours_t50_params.values()
	
	# Rescale the params such that largest is 100 and rest are scaled accordingly
	max_params = max(max(params_10t), max(params_50t))
	
	# Set font size
	plt.rcParams.update({'font.size': FONT_SIZE})  # Change 12 to the desired font size for A4 PDF
	
	# Plotting the performance plots
	fig, ax1 = plt.subplots(figsize=(8, 6))
	
	# Evenly spaced x-axis values
	x_values = np.arange(len(ranks))
	
	ax1.plot(x_values, perf_cruxI_10t, marker='o', label='I, m=10', color=colors[0])
	ax1.plot(x_values, perf_cruxO_10t, marker='s', label='O, m=10', color=colors[1])
	ax1.plot(x_values, perf_cruxI_50t, marker='^', label='I, m=50', color=colors[2])
	ax1.plot(x_values, perf_cruxO_50t, marker='x', label='O, m=50', color=colors[3])
	
	ax1.set_xticks(x_values)
	ax1.set_xticklabels(ranks)
	
	# Adding labels and legend for performance plots
	ax1.set_xlabel('Rank (r)')
	ax1.set_ylabel('pass@1', color='black')
	ax1.tick_params(axis='y', labelcolor='black')
	# ax1.set_title('Performance vs Rank')
	
	# Displaying legend for performance plots below the plot
	# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
	
	# Plotting the bar plots for number of parameters
	ax2 = ax1.twinx()
	bar_width = 0.2
	
	# Adjusting the position of bars for
	bar_pos_1 = x_values
	bar_pos_2 = x_values + bar_width
	
	# Bar plots for number of parameters
	ax2.bar(bar_pos_1, params_10t, bar_width, label='Params(m=10)', color=colors[4], alpha=0.2)
	ax2.bar(bar_pos_2, params_50t, bar_width, label='Params(m=50)', color=colors[4], alpha=0.4)
	
	# Adding labels and legend for number of parameters
	ax2.set_ylabel('Parameters (in millions)', color='black')
	ax2.tick_params(axis='y', labelcolor='black')
	
	# ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=2)
	
	# Adding grid lines to both axes
	ax1.grid(True)
	ax2.grid(False)  # Don't show grid lines for number of parameters
	
	# Adjust layout to prevent overlap of labels
	plt.tight_layout()
	
	# Save the plot in high resolution
	plt.savefig(f'plots/ablation_rank_on_crux_w_param.png', dpi=500)  # Change the file format and dpi as needed
	

if __name__ == '__main__':
	
	# baseline_perf_w_param_comparison()
	
	ablation_on_rank_glue_w_param()
	
	ablation_on_rank_w_param_cruxeval()

	
