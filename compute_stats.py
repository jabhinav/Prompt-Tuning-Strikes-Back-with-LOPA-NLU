"""
Compute statistics for the data. Average percentage improvement between ours and baseline
"""


baseline_pt = [
	84.40, 54.67, 72.38, 58.74, 48.20, 53.07,  # NLU
	32.8, 41.2, 35.0, 33.5, 45.8, 37.5,  # CRUXEval-I
	15.8, 31.2, 34.0, 31.5, 44.8, 32.0,  # CRUXEval-O
	15.20, 34.49, 49.69, 34.08, 37.47, 24.23  # MBPP
]

baseline_idpg = [
	95.30, 84.50, 78.60, 90.48, 84.88, 77.26,  # NLU
	16.9, 26.0, 35.0, 31.0, 40.5, 29.2,  # CRUXEval-I
	13.2, 28.5, 33.0, 39.5, 41.5, 35.2,  # CRUXEval-O
	17.04, 42.50, 53.29, 42.29, 53.59, 32.44  # MBPP
]

baseline_lora = [
	96.22, 90.30, 90.77, 94.69, 89.91, 85.66,  # NLU
	31.0, 35.5, 41.5, 38.0, 47.5, 45.5,  # CRUXEval-I
	18.2, 36.0, 42.5, 41.2, 49.8, 40.5,  # CRUXEval-O
	21.56, 44.14, 51.54, 42.92, 53.38, 44.55  # MBPP
]

baseline_ours = [
	95.99, 89.22, 91.09, 93.74, 89.72, 83.39,  # NLU
	34.5, 43.0, 43.0, 42.2, 50.0, 41.2,  # CRUXEval-I
	18.5, 35.0, 38.2, 42.5, 48.0, 39.8,  # CRUXEval-O
	17.04, 44.66, 52.15, 44.35, 52.46, 43.94  # MBPP
]


def get_avg_percent_improvement(perf1, perf2):
	avg_percent_improvement = 0
	for p1, p2 in zip(perf1, perf2):
		avg_percent_improvement += (p1 - p2) / p2  # (ours - baseline) / baseline
	avg_percent_improvement /= len(perf1)
	return avg_percent_improvement


def get_number_of_wins(perf1, perf2):
	wins = 0
	for p1, p2 in zip(perf1, perf2):
		if p1 > p2:
			wins += 1
	return wins

num_evals = len(baseline_ours)

# Ours vs PT
avg_percent_improvement = get_avg_percent_improvement(baseline_ours, baseline_pt)
wins = get_number_of_wins(baseline_ours, baseline_pt)
print(f"Ours Average % Improvement over PT: {avg_percent_improvement * 100:.2f}%, Wins: {wins}/{num_evals}")

# Ours vs IDPG
avg_percent_improvement = get_avg_percent_improvement(baseline_ours, baseline_idpg)
wins = get_number_of_wins(baseline_ours, baseline_idpg)
print(f"Ours Average % Improvement over IDPG: {avg_percent_improvement * 100:.2f}%, Wins: {wins}/{num_evals}")

# LORA vs Ours
avg_percent_improvement = get_avg_percent_improvement(baseline_lora, baseline_ours)
wins = get_number_of_wins(baseline_lora, baseline_ours)
print(f"LORA Average % Improvement over Ours: {avg_percent_improvement * 100:.2f}%, Wins: {wins}/{num_evals}")

# Ours vs LORA
avg_percent_improvement = get_avg_percent_improvement(baseline_ours, baseline_lora)
wins = get_number_of_wins(baseline_ours, baseline_lora)
print(f"Ours Average % Improvement over LORA: {avg_percent_improvement * 100:.2f}%, Wins: {wins}/{num_evals}")