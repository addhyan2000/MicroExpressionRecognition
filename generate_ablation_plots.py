import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Figures directory
fig_dir = Path("c:/Users/Addhyan/Desktop/Thesis/Thesis3/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# Data definition for key configurations
configs = [
    "Config 1\n(Pure Base)",
    "Config 2\n(Temporal)",
    "Config 3\n(Spatial)",
    "Config 4\n(EVM Base)",
    "Config 5\n(Attention)",
    "Config 7\n(Full no Attn)",
    "Config 8\n(Proposed)"
]

accuracy = [74.36, 74.36, 71.79, 74.36, 58.97, 69.23, 48.72]
macro_f1 = [28.43, 28.43, 43.57, 28.43, 40.52, 35.77, 30.94]

neg_f1 = [85.29, 85.29, 80.70, 85.29, 69.39, 80.65, 58.33]
pos_f1 = [0.0, 0.0, 50.00, 0.0, 52.17, 26.67, 34.48]
sur_f1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# --- Plot 1: Performance Comparison (Accuracy vs Macro F1) ---
plt.figure(figsize=(10, 6))
x = np.arange(len(configs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#2b6cb0')
rects2 = ax.bar(x + width/2, macro_f1, width, label='Macro F1 (%)', color='#dd6b20')

ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study: Accuracy vs Macro F1 Score across Key Configurations', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig(fig_dir / "ablation_performance_comparison.png", dpi=300)
plt.close()
print("Generated ablation_performance_comparison.png")

# --- Plot 2: Per-Class F1 Score Comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
width2 = 0.25
rects_neg = ax.bar(x - width2, neg_f1, width2, label='Negative F1 (%)', color='#e53e3e')
rects_pos = ax.bar(x, pos_f1, width2, label='Positive F1 (%)', color='#38a169')
rects_sur = ax.bar(x + width2, sur_f1, width2, label='Surprise F1 (%)', color='#3182ce')

ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study: Per-Class F1-Score comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

autolabel(rects_neg)
autolabel(rects_pos)
# For surprise, only label non-zero values
for rect in rects_sur:
    height = rect.get_height()
    if height > 0:
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / "ablation_class_f1_comparison.png", dpi=300)
plt.close()
print("Generated ablation_class_f1_comparison.png")
