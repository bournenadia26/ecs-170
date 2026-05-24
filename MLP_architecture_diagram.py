import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis('off')

# Draw layers (shift all x positions right by 0.05)
ax.text(0.10, 0.5, 'Input\n(784)', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='lightblue'))
ax.arrow(0.17, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='k', ec='k', length_includes_head=True)
ax.text(0.27, 0.5, 'Dense\n512', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='wheat'))
ax.arrow(0.34, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='k', ec='k', length_includes_head=True)
ax.text(0.44, 0.5, 'ReLU', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen'))
ax.arrow(0.51, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='k', ec='k', length_includes_head=True)
ax.text(0.61, 0.5, 'Dropout\n0.3', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='lightgray'))
ax.arrow(0.68, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='k', ec='k', length_includes_head=True)
ax.text(0.78, 0.5, 'Dense\n256', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='wheat'))
ax.arrow(0.85, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='k', ec='k', length_includes_head=True)
ax.text(0.95, 0.5, 'ReLU', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen'))
ax.arrow(1.02, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='k', ec='k', length_includes_head=True)
ax.text(1.12, 0.5, 'Dropout\n0.3', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='lightgray'))
ax.arrow(1.19, 0.5, 0.08, 0, head_width=0.05, head_length=0.02, fc='k', ec='k', length_includes_head=True)
ax.text(1.29, 0.5, 'Output\n(10)', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='salmon'))

# Add extra left margin and padding
plt.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.15)
plt.savefig('MLP_architecture.png', dpi=200, bbox_inches='tight', pad_inches=0.3)
plt.close()
