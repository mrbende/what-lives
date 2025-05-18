import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle

def create_publication_chart(figsize=(9, 5), save_path=None, dpi=600):
    # Set up custom style for publication
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'xtick.major.width': 0.6,
        'xtick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.major.width': 0,
        'text.color': '#333333',
        'axes.labelcolor': '#333333'
    })
    
    # Create figure with specific dimensions
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Data (percentages and labels)
    categories = ['Perspective', 'Nature', 'Approach']
    left_values = [57, 46, 87.7]
    right_values = [43, 54, 12.3]
    left_labels = ['Objective', 'Binary', 'Actionable (conventionally defined)']
    right_labels = ['Observer-relative', 'Continuous property', 'Inspirational']
    
    # Color palette (professionally selected, color-blind friendly)
    # Base colors with more sophisticated shades
    left_colors = ['#4568b0', '#dd7631', '#52a447']  # Deeper, richer primary colors
    right_colors = ['#92b4f4', '#f6d0a9', '#b7e0a8']  # Lighter, harmonious secondary colors
    
    # Calculate positions
    y_positions = np.arange(len(categories)) * 1.5
    height = 0.55
    
    # Add subtle background grid
    ax.grid(axis='x', linestyle='-', alpha=0.1, zorder=0)
    
    # Plot each category (in reverse order to match the image)
    for i, (category, left_val, right_val, left_label, right_label, left_color, right_color) in enumerate(
        zip(reversed(categories), reversed(left_values), reversed(right_values), 
            reversed(left_labels), reversed(right_labels), 
            reversed(left_colors), reversed(right_colors))):
        
        # Calculate y-position (reversed)
        y_pos = i
        
        # Plot bars
        ax.barh(y_pos, left_val, height=height, color=left_color, edgecolor='white', linewidth=0.5, zorder=2)
        ax.barh(y_pos, right_val, height=height, left=left_val, color=right_color, edgecolor='white', linewidth=0.5, zorder=2)
        
        # Add percentage text (with smart contrast adjustment)
        left_text_color = 'white' if left_val > 25 else '#333333'
        right_text_color = 'white' if right_val > 25 else '#333333'
        
        # Center percentage labels within their sections
        ax.text(left_val/2, y_pos, f"{left_val}%", 
                ha='center', va='center', color=left_text_color, 
                fontweight='bold', fontsize=12, zorder=5)
        
        ax.text(left_val + right_val/2, y_pos, f"{right_val}%", 
                ha='center', va='center', color=right_text_color, 
                fontweight='bold', fontsize=12, zorder=5)
        
        # Category label
        ax.text(-5, y_pos, category, 
                ha='right', va='center', fontsize=14, fontweight='bold', color='#333333')
        
        # Option labels (positioned better with consistent spacing)
        label_y_offset = 0.4
        ax.text(left_val/2, y_pos + label_y_offset, left_label, 
                ha='center', va='center', fontsize=12, color='#333333')
        
        ax.text(left_val + right_val/2, y_pos + label_y_offset, right_label, 
                ha='center', va='center', fontsize=12, color='#333333')
    
    # Configure axis for publication style
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel('Percentage (%)', fontsize=14, labelpad=7, color='#333333')
    
    # Add padding at the top for label clarity
    ax.set_ylim(-0.5, len(categories)-0.5 + 0.33)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig, ax

# Example usage
# if __name__ == "__main__":
# # Create basic chart
fig, ax = create_publication_chart()

# Optional: Save as vector for publication (best quality)
# create_publication_chart(save_path="publication_chart.pdf")
create_publication_chart(save_path="/workspace/what-lives/data/publication_chart.png", dpi=600)

plt.show()