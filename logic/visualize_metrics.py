import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for professional-looking plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Read the CSV files
df_4o = pd.read_csv('4o_verification_metrics.csv')
df_4o_mini = pd.read_csv('4o_mini_verification_metrics.csv')

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# 1. Average Inference Time Comparison
avg_times_4o = df_4o.groupby('Task Name')['Inference Time (s)'].mean()
avg_times_4o_mini = df_4o_mini.groupby('Task Name')['Inference Time (s)'].mean()

x = np.arange(len(avg_times_4o))
width = 0.35

ax1.bar(x - width/2, avg_times_4o.values, width, label='4o Model', color='#2ecc71')
ax1.bar(x + width/2, avg_times_4o_mini.values, width, label='4o Mini Model', color='#3498db')

ax1.set_ylabel('Average Inference Time (s)', fontsize=12)
ax1.set_title('Average Inference Time by Task', fontsize=14, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(avg_times_4o.index, rotation=45)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Human Marked Correct Comparison
correct_4o = df_4o.groupby('Task Name')['Human Marked Correct'].apply(lambda x: (x == 'Yes').sum())
correct_4o_mini = df_4o_mini.groupby('Task Name')['Human Marked Correct'].apply(lambda x: (x == 'Yes').sum())

ax2.bar(x - width/2, correct_4o.values, width, label='4o Model', color='#2ecc71')
ax2.bar(x + width/2, correct_4o_mini.values, width, label='4o Mini Model', color='#3498db')

ax2.set_ylabel('Number of Correct Verifications', fontsize=12)
ax2.set_title('Number of Human-Verified Correct Results', fontsize=14, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(correct_4o.index, rotation=45)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Accuracy Rate Comparison (Correct / Total)
total_4o = df_4o.groupby('Task Name').size()
total_4o_mini = df_4o_mini.groupby('Task Name').size()

accuracy_4o = (correct_4o / total_4o) * 100
accuracy_4o_mini = (correct_4o_mini / total_4o_mini) * 100

ax3.bar(x - width/2, accuracy_4o.values, width, label='4o Model', color='#2ecc71')
ax3.bar(x + width/2, accuracy_4o_mini.values, width, label='4o Mini Model', color='#3498db')

ax3.set_ylabel('Accuracy Rate (%)', fontsize=12)
ax3.set_title('Verification Accuracy Rate by Task', fontsize=14, pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(accuracy_4o.index, rotation=45)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('verification_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close() 