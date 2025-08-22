import random 
import gzip
import os 
from scipy import interpolate
from scipy.signal import convolve2d
from pdb import set_trace as st 
import json
import numpy as np
import matplotlib.pyplot as plt

def save_json(description, output_dir, idx):
    with open(f"{output_dir}/labels/{idx:06d}.json", 'w') as f:
        json.dump(description, f, indent=2)  # indent=2 for human readability
        

def simulate_handwritten_line(start_point, end_point, num_points=200, jitter_strength=0.05,smooth = np.random.randint(5,10)):
    """
    Generates a simulated handwritten line between two points.

    Args:
        start_point (tuple): (x, y) coordinates of the starting point.
        end_point (tuple): (x, y) coordinates of the ending point.
        num_points (int): The number of points to generate along the line.
        jitter_strength (float): Controls the amount of randomness.

    Returns:
        tuple: A tuple of two NumPy arrays (x_coords, y_coords).
    """
    # 1. Generate a smooth, straight path
    x_coords = np.linspace(start_point[0], end_point[0], num_points)
    y_coords = np.linspace(start_point[1], end_point[1], num_points)

    # 2. Add random jitter to the coordinates
    # np.random.randn() generates random values from a standard normal distribution
    x_jitter = np.random.randn(num_points) * jitter_strength
    y_jitter = np.random.randn(num_points) * jitter_strength
    
    x_simulated = x_coords + x_jitter
    y_simulated = y_coords + y_jitter
    v = np.ones(smooth)/smooth
    x_simulated = np.convolve(x_simulated,v)[smooth:-smooth]
    y_simulated = np.convolve(y_simulated,v)[smooth:-smooth]
    return x_simulated, y_simulated

def format_length_with_sqrt(length, tolerance=1e-10):
    """
    Format length as square root if it's the square root of an integer.
    Returns a tuple: (formatted_string, is_sqrt_form)
    """
    # Check if the length squared is close to an integer
    length_squared = length ** 2
    nearest_int = round(length_squared)
    
    if abs(length_squared - nearest_int) < tolerance and nearest_int > 0:
        # Check if the length itself is not already an integer
        if abs(length - round(length)) > tolerance:
            return f'√{nearest_int}', True
        else:
            # Length is already an integer, return as is
            return str(int(round(length))), False
    else:
        # Not a perfect square root, return rounded decimal
        return str(round(length, 2)), False

def draw_shortest_arc(start_point, center, end_point, num_points=100):
    """Draw an arc from start_point to end_point around center"""
    start_vec = start_point - center
    end_vec = end_point - center
    
    start_angle = np.arctan2(start_vec[1], start_vec[0])
    end_angle = np.arctan2(end_vec[1], end_vec[0])
    
    radius = np.linalg.norm(start_vec)
    
    # Ensure we take the shorter arc
    angle_diff = end_angle - start_angle
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    angles = np.linspace(start_angle, start_angle + angle_diff, num_points)
    arc_x = center[0] + radius * np.cos(angles)
    arc_y = center[1] + radius * np.sin(angles)
    
    plt.plot(arc_x, arc_y, 'black', linewidth=1.5)


def draw_handwritten_arc(start_point, center, end_point, jitter_amount=0.02, 
                        smoothing_factor=0.2, num_points=100, line_width=1.5, 
                        alpha=0.9, color='black', ax=None):

    if ax is None:
        ax = plt.gca()
    
    # Convert to numpy arrays
    start_point = np.array(start_point)
    center = np.array(center)
    end_point = np.array(end_point)
    
    # Calculate vectors and angles
    start_vec = start_point - center
    end_vec = end_point - center
    
    start_angle = np.arctan2(start_vec[1], start_vec[0])
    end_angle = np.arctan2(end_vec[1], end_vec[0])
    
    radius = np.linalg.norm(start_vec)
    
    # Ensure we take the shorter arc
    angle_diff = end_angle - start_angle
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # Generate base arc points
    angles = np.linspace(start_angle, start_angle + angle_diff, num_points)
    arc_x = center[0] + radius * np.cos(angles)
    arc_y = center[1] + radius * np.sin(angles)
    
    # Add jitter - less jitter for arcs to maintain circular shape
    jitter_scale = jitter_amount * radius
    jitter_x = np.random.normal(0, jitter_scale, num_points)
    jitter_y = np.random.normal(0, jitter_scale, num_points)
    
    x_jittered = arc_x + jitter_x
    y_jittered = arc_y + jitter_y
    
    # Apply smoothing - important for maintaining arc-like shape
    if smoothing_factor > 0:
        try:
            tck, u = interpolate.splprep([x_jittered, y_jittered], s=smoothing_factor * num_points)
            x_smooth, y_smooth = interpolate.splev(np.linspace(0, 1, num_points), tck)
        except:
            # Fallback if spline fails
            x_smooth, y_smooth = x_jittered, y_jittered
    else:
        x_smooth, y_smooth = x_jittered, y_jittered
    
    # Vary line width to simulate pen pressure
    t = np.linspace(0, 1, num_points)
    line_widths = line_width * (0.7 + 0.6 * np.sin(np.pi * t))
    
    # Draw the main line
    line = ax.plot(x_smooth, y_smooth, color=color, alpha=alpha, 
                  linewidth=line_widths[0])[0]
    
    # Add subtle ink bleed effect
    for i in range(1, 2):  # Fewer layers for arcs
        offset = line_width * 0.08 * i
        ax.plot(x_smooth + offset * np.cos(angles), 
                y_smooth + offset * np.sin(angles), 
                color=color, alpha=alpha*0.2, linewidth=line_widths[0]*0.4)





def smart_label_offset(x, y, vertices, offset_distance=0.2):
    """Calculate smart offset for vertex labels to avoid overlaps"""
    # Convert vertices to array for easier calculation
    coords = np.array([[v["x"], v["y"]] for v in vertices.values()])
    current_point = np.array([x, y])
    
    # Find the direction that has the most space
    directions = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1], 
                          [1, 0], [-1, 0], [0, 1], [0, -1]])
    
    best_direction = directions[0]
    max_min_distance = 0
    
    for direction in directions:
        test_point = current_point + direction * offset_distance
        # Calculate minimum distance to all other points
        distances = np.linalg.norm(coords - test_point, axis=1)
        distances = distances[distances > 0]  # Remove distance to itself
        
        if len(distances) > 0:
            min_distance = np.min(distances)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_direction = direction
    
    return current_point + best_direction * offset_distance



def draw_mark(xs,ys):
    vector = [np.diff(xs)[0],np.diff(ys)[0]]
    vector = np.array([vector[1],-vector[0]])
    vector_size = np.sqrt((vector**2).sum())
    vector *= 0.5/vector_size
    plt.plot([xs.mean()-vector[0],xs.mean()+vector[0]],[ys.mean()-vector[1],ys.mean()+vector[1]],color = 'black')



def read_idx_ubyte(file_path):
    """
    Parses a .ubyte file from the MNIST/EMNIST dataset.
    Returns the data as a numpy array.
    """
    # Check if the file is gzipped or already unzipped
    if file_path.endswith('.gz'):
        file_handle = gzip.open(file_path, 'rb')
    else:
        file_handle = open(file_path, 'rb')

    with file_handle as f:
        # Read the magic number and dimensions from the header
        magic_number = int.from_bytes(f.read(4), 'big')
        num_items = int.from_bytes(f.read(4), 'big')

        # Check if it's an image file (magic number 2051)
        if magic_number == 2051:
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            print(f"Reading image data: {num_items} images of {rows}x{cols} pixels.")
            # Read the pixel data and reshape it
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)

        # Check if it's a label file (magic number 2049)
        elif magic_number == 2049:
            print(f"Reading label data: {num_items} labels.")
            # Read the label data
            data = np.frombuffer(f.read(), dtype=np.uint8)

        else:
            raise ValueError(f"Invalid magic number in file: {magic_number}")

    return data



def create_handwritten_dot():
    # Create a blank 28x28 canvas (black background)
    dot_mat = np.zeros((28, 12))

    # Define the core size of the dot
    dot_size = np.random.choice([2,3])
    # Add a small random offset to the center to simulate the natural
    # variation in handwriting
    center_x = 6 + random.randint(-1, 1)
    center_y = 22 + random.randint(-2, 2) # Place the dot near the bottom of the grid

    # Create the base dot shape. A small filled circle works well.
    for i in range(28):
        for j in range(28):
            if (i - center_y)**2 + (j - center_x)**2 < dot_size**2:
                dot_mat[i, j] = 1.0

    # Define a simple blurring kernel. A small kernel of ones will act as a
    # blurring filter, smudging the dot's edges.
    kernel_size = np.random.choice([2,3])
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    # Perform the 2D convolution
    convolved_dot = convolve2d(dot_mat, kernel, mode='same', boundary='symm')

    # Normalize the convolved matrix to the range [0, 1] for proper display
    convolved_dot = convolved_dot / np.max(convolved_dot) if np.max(convolved_dot) > 0 else convolved_dot

    return convolved_dot




