import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set up IEEE/ISCA/MICRO aesthetic requirements
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#E0E0E0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

# ColorBrewer Dark2 palette
C_MXFP4 = '#1B9E77'
C_M2XFP4 = '#D95F02'
C_MXFP8 = '#7570B3'
C_MXFP4R = '#E7298A'

# Data: Useful Throughput (Table VI)
bot_labels = ['MXFP4', 'M$^2$XFP4', 'MXFP8', 'MXFP4R']
bot_values = [13.28, 11.50, 4.61, 14.11]
bot_colors = [C_MXFP4, C_M2XFP4, C_MXFP8, C_MXFP4R]
bot_speedups = ['2.88$\\times$', '2.49$\\times$', '1.00$\\times$', '3.06$\\times$']

# Figure Setup - Single Panel matching Table V height
fig, ax = plt.subplots(figsize=(3.3, 2.2), dpi=300)

ax.grid(True, axis='y', color='#E0E0E0', linestyle='-', linewidth=0.5, zorder=0)
ax.grid(False, axis='x')

bars = ax.bar(np.arange(len(bot_labels)), bot_values, color=bot_colors, 
              edgecolor='black', linewidth=0.5, width=0.55, zorder=3)
ax.set_xticks(np.arange(len(bot_labels)))
ax.set_xticklabels(bot_labels)
ax.set_ylabel('Throughput (GFLOP/s)')
ax.set_title('Useful Throughput vs. MXFP8', pad=12, fontweight='bold')

# Annotate values and speedups
for i, bar in enumerate(bars):
    height = bar.get_height()
    # Value label
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.4, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=7.5, color='black')
    # Speedup label (bold)
    ax.text(bar.get_x() + bar.get_width()/2, height + 1.8, bot_speedups[i], 
            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='black')

ax.set_ylim(0, max(bot_values) * 1.35)

plt.tight_layout()
plt.savefig('mxfp4r_throughput_comparison.png', bbox_inches='tight', dpi=300)
print('Updated single-panel figure generated successfully.')
