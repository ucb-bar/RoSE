import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_trace_file(file_content):
    # Split content into lines and skip warmup
    lines = [line.strip() for line in file_content.split('\n') if line.strip() and not line.startswith('warmup')]
    
    # Parse each line into a dictionary
    data = []
    for line in lines:
        parts = line.split(',')
        vq = float(parts[0].split(': ')[1])
        time = int(parts[1].split(': ')[1])
        path = parts[2].split(':')[1].split('-')[1]
        data.append({'vq': vq, 'time': time, 'path': path})
    
    return pd.DataFrame(data)

def plot_circular_paths(df):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    theta = df['vq']
    
    # Get unique paths and assign colors
    unique_paths = df['path'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_paths)))
    path_colors = dict(zip(unique_paths, colors))
    
    # First subplot: Circular path visualization
    circle = plt.Circle((0, 0), 0.8, fill=False, color='black', alpha=0.3)
    ax1.add_artist(circle)
    
    # Plot paths
    for path in unique_paths:
        mask = df['path'] == path
        r = np.ones(mask.sum()) * 0.8
        theta_path = theta[mask]
        
        x = r * np.cos(theta_path)
        y = r * np.sin(theta_path)
        # Use same label for both plots
        label = f'Path: {path[:10]}...'
        ax1.scatter(x, y, c=[path_colors[path]], s=100, alpha=0.7, label=label)
        ax2.scatter(theta[mask], df['time'][mask], c=[path_colors[path]], s=100, alpha=0.7)
    
    # Add labels for 0 and π
    ax1.text(0.9, 0, '0', color='black', fontsize=12, ha='left', va='center')
    ax1.text(-0.9, 0, 'π', color='black', fontsize=12, ha='right', va='center')
    
    ax1.set_aspect('equal')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_title('Sin() Function Execution Paths\nColor represents different branch paths')
    
    ax2.set_xlabel('Input angle (radians)')
    ax2.set_ylabel('Execution time')
    ax2.set_title('Execution Time per Input')
    ax2.grid(True, alpha=0.3)
    
    # Single legend outside both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.5))
    
    # Remove axes from circular plot
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add subtle grid in polar coordinates for circular plot
    for r in np.linspace(0.2, 0.8, 4):
        circle = plt.Circle((0, 0), r, fill=False, color='gray', alpha=0.1)
        ax1.add_artist(circle)
    
    for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = [0, np.cos(theta)]
        y = [0, np.sin(theta)]
        ax1.plot(x, y, color='gray', alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('foc.png', bbox_inches='tight')
    plt.show()

# Read the trace data
with open('trace.foc.txt', 'r') as file:
    trace_data = file.read()

# Create visualization
df = parse_trace_file(trace_data)
plot_circular_paths(df)