import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

# Set publication quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

# Read the CSV files
df_4o = pd.read_csv('4o_verification_metrics.csv')
df_4o_mini = pd.read_csv('4o_mini_verification_metrics.csv')

# Token usage data
token_usage = {
    '4o Model': 16000,
    '4o Mini Model': 610000
}

# Print average accuracy and inference time for both models
avg_accuracy_4o = df_4o['Human Marked Correct'].eq('Yes').mean() * 100
avg_accuracy_4o_mini = df_4o_mini['Human Marked Correct'].eq('Yes').mean() * 100
avg_time_4o = df_4o['Inference Time (s)'].mean()
avg_time_4o_mini = df_4o_mini['Inference Time (s)'].mean()

print(f"4o Model: Average Accuracy = {avg_accuracy_4o:.2f}% | Average Inference Time = {avg_time_4o:.2f} s")
print(f"4o Mini Model: Average Accuracy = {avg_accuracy_4o_mini:.2f}% | Average Inference Time = {avg_time_4o_mini:.2f} s")

# Figure 1: Performance Metrics Overview
fig1 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig1, height_ratios=[1, 1.2])

# 1.1 Inference Time Box Plot
ax1 = fig1.add_subplot(gs[0, 0])
sns.boxplot(data=pd.concat([
    df_4o.assign(Model='4o Model'),
    df_4o_mini.assign(Model='4o Mini Model')
]), x='Task Name', y='Inference Time (s)', hue='Model', ax=ax1, palette=['#2ecc71', '#3498db'])
ax1.set_title('Inference Time Distribution', pad=20)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_ylabel('Time (seconds)')
ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# 1.2 Accuracy Comparison
ax2 = fig1.add_subplot(gs[0, 1])
accuracy_data = pd.DataFrame({
    'Task': df_4o['Task Name'].unique(),
    '4o Model': df_4o.groupby('Task Name')['Human Marked Correct'].apply(lambda x: (x == 'Yes').mean() * 100),
    '4o Mini Model': df_4o_mini.groupby('Task Name')['Human Marked Correct'].apply(lambda x: (x == 'Yes').mean() * 100)
})
accuracy_data.plot(x='Task', y=['4o Model', '4o Mini Model'], kind='bar', ax=ax2, color=['#2ecc71', '#3498db'])
ax2.set_title('Verification Accuracy by Task', pad=20)
ax2.set_ylabel('Accuracy (%)')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# 1.3 Token Usage vs Performance
ax3 = fig1.add_subplot(gs[1, :])
avg_accuracy = {
    '4o Model': df_4o['Human Marked Correct'].eq('Yes').mean() * 100,
    '4o Mini Model': df_4o_mini['Human Marked Correct'].eq('Yes').mean() * 100
}
avg_time = {
    '4o Model': df_4o['Inference Time (s)'].mean(),
    '4o Mini Model': df_4o_mini['Inference Time (s)'].mean()
}

scatter = ax3.scatter(
    [token_usage['4o Model'], token_usage['4o Mini Model']],
    [avg_accuracy['4o Model'], avg_accuracy['4o Mini Model']],
    s=[avg_time['4o Model']*200, avg_time['4o Mini Model']*200],
    c=['#2ecc71', '#3498db'],
    alpha=0.7,
    label=['4o Model', '4o Mini Model']
)

ax3.set_xscale('log')
ax3.set_xlabel('Token Usage (log scale)')
ax3.set_ylabel('Average Accuracy (%)')
ax3.set_title('Token Usage vs Performance Trade-off', pad=20)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
ax3.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add annotations with arrows
for model in ['4o Model', '4o Mini Model']:
    ax3.annotate(f'{avg_accuracy[model]:.1f}% accuracy\n{avg_time[model]:.1f}s avg time',
                (token_usage[model], avg_accuracy[model]),
                xytext=(20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

plt.tight_layout()
plt.savefig('performance_overview.png')
plt.close()

# Figure 2: Detailed Task Analysis
fig2 = plt.figure(figsize=(15, 12))
gs2 = GridSpec(2, 1, figure=fig2, height_ratios=[1.2, 1])

# 2.1 Task-wise Performance Metrics
ax4 = fig2.add_subplot(gs2[0])
task_metrics = pd.DataFrame({
    'Task': df_4o['Task Name'].unique(),
    '4o_Accuracy': df_4o.groupby('Task Name')['Human Marked Correct'].apply(lambda x: (x == 'Yes').mean() * 100),
    '4o_Mini_Accuracy': df_4o_mini.groupby('Task Name')['Human Marked Correct'].apply(lambda x: (x == 'Yes').mean() * 100),
    '4o_Time': df_4o.groupby('Task Name')['Inference Time (s)'].mean(),
    '4o_Mini_Time': df_4o_mini.groupby('Task Name')['Inference Time (s)'].mean()
})

x = np.arange(len(task_metrics['Task']))
width = 0.35

ax4_twin = ax4.twinx()
bars1 = ax4.bar(x - width/2, task_metrics['4o_Accuracy'], width, label='4o Accuracy', color='#2ecc71', alpha=0.7)
bars2 = ax4.bar(x + width/2, task_metrics['4o_Mini_Accuracy'], width, label='4o Mini Accuracy', color='#3498db', alpha=0.7)
line1 = ax4_twin.plot(x - width/2, task_metrics['4o_Time'], 'o-', label='4o Time', color='#e74c3c', linewidth=3, markersize=10, marker='o')
line2 = ax4_twin.plot(x + width/2, task_metrics['4o_Mini_Time'], 's--', label='4o Mini Time', color='#8e44ad', linewidth=3, markersize=10, marker='s')

ax4.set_ylabel('Accuracy (%)')
ax4_twin.set_ylabel('Average Time (s)')
ax4.set_title('Task-wise Performance Analysis', pad=20)
ax4.set_xticks(x)
ax4.set_xticklabels(task_metrics['Task'], rotation=45, ha='right')
ax4.yaxis.set_major_formatter(mtick.PercentFormatter())

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))

# 2.2 Efficiency Analysis
ax5 = fig2.add_subplot(gs2[1])
efficiency_data = pd.DataFrame({
    'Task': df_4o['Task Name'].unique(),
    '4o_Efficiency': df_4o.groupby('Task Name').apply(lambda x: (x['Human Marked Correct'] == 'Yes').mean() / x['Inference Time (s)'].mean()),
    '4o_Mini_Efficiency': df_4o_mini.groupby('Task Name').apply(lambda x: (x['Human Marked Correct'] == 'Yes').mean() / x['Inference Time (s)'].mean())
})

efficiency_data.plot(x='Task', y=['4o_Efficiency', '4o_Mini_Efficiency'], kind='bar', ax=ax5, color=['#2ecc71', '#3498db'])
ax5.set_title('Task Efficiency (Accuracy per Second)', pad=20)
ax5.set_ylabel('Efficiency Score')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
ax5.legend(title='Model', bbox_to_anchor=(1.15, 0.5))

plt.tight_layout()
plt.savefig('detailed_analysis.png')
plt.close()

# Figure 3: Token Usage Analysis
fig3 = plt.figure(figsize=(15, 6))
gs3 = GridSpec(1, 2, figure=fig3, width_ratios=[1, 1.2])

# 3.1 Token Usage Comparison
ax6 = fig3.add_subplot(gs3[0])
wedges, texts, autotexts = ax6.pie([token_usage['4o Model'], token_usage['4o Mini Model']], 
        labels=['4o Model', '4o Mini Model'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#3498db'],
        textprops={'fontsize': 12})
ax6.set_title('Token Usage Distribution', pad=20)

# 3.2 Token Efficiency
ax7 = fig3.add_subplot(gs3[1])
token_efficiency = {
    '4o Model': avg_accuracy['4o Model'] / token_usage['4o Model'] * 1000,
    '4o Mini Model': avg_accuracy['4o Mini Model'] / token_usage['4o Mini Model'] * 1000
}

bars = ax7.bar(['4o Model', '4o Mini Model'], 
        [token_efficiency['4o Model'], token_efficiency['4o Mini Model']],
        color=['#2ecc71', '#3498db'])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

ax7.set_title('Token Efficiency (Accuracy per 1000 Tokens)', pad=20)
ax7.set_ylabel('Efficiency Score')

plt.tight_layout()
plt.savefig('token_analysis.png')
plt.close() 