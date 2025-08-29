import string
import random 
import gzip
import os 
from scipy import interpolate
from scipy.signal import convolve2d
from pdb import set_trace as st 
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
import textwrap 
from scipy.ndimage import rotate,zoom

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
    coords = np.array([[v["x"], v["y"]] for v in vertices])
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



class EMNIST_Handler():
    def __init__(self,emnist_images,emnist_labels,
            emnist_chars = string.digits+string.ascii_uppercase+string.ascii_lowercase):
        self.emnist_images = emnist_images
        self.emnist_labels = emnist_labels
        self.emnist_chars = emnist_chars
        
    def char_mat(self,ch):
        ind = np.random.choice(np.where(self.emnist_labels == self.emnist_chars.find(ch))[0])
        mat = 1-(self.emnist_images[ind].T/255)            
        return mat
    def horizontal_line(self):
        mat = 1-self.char_mat("1")
        line = np.zeros([28,28])
        line[:,12:16] = 1 
        alphas = np.arange(-90,90,1)
        y = alphas*0
        def convolve(image,image2,alpha):
            return ((1-image)*rotate(image2, angle=alpha, reshape=False, mode='constant', cval=0)).sum()
        for i,alpha in enumerate(alphas):
            y[i] = convolve(mat,line,alpha)    # Rotate the original image to align it vertically
        aligned_image = rotate(mat, angle=-alphas[y.argmin()], reshape=False, mode='constant', cval=0)
        return 1-aligned_image.T
    def equal_sign(self):
        mat = self.horizontal_line()
        return np.concatenate([mat[4:-10],mat[10:-4]],axis = 0)
    def angle_sign(self):
        line = 1-self.horizontal_line().T
        limb = rotate(line,angle = 45,reshape=False,mode='constant',cval = 0)
        edges = self.find_edges(limb)
        limb = limb[edges[0]+2:edges[1]+2,edges[2]-2:edges[3]-2]
        limb = zoom(limb,(28/limb.shape[0],14/limb.shape[1]))
        limb = np.concatenate([np.zeros([5,14]),limb[5:]],axis = 0)
        angle = np.concatenate([limb,limb[:,::-1]],axis = 1)[::-1].T+line
        angle[angle>1] = 1
        return 1-angle
    def find_edges(self,mat):
        width_pixels = np.where(mat.sum(axis = 0) > 0.1)[0]
        left = width_pixels.min()
        right = width_pixels.max()
        height_pixels = np.where(mat.sum(axis = 1) > 0.1)[0]
        up = height_pixels.min()
        down = height_pixels.max()
        return up,down,left,right
    def token_mat(self,token):
        #### input is some token - word, sentence, terms ("AB = 5"), returns same token handwritten.
        mats = []
        for char in token:
            if char == ' ':
                mats.append(np.ones([28,14]))
            elif char == '=':
                mats.append(self.equal_sign())
            elif char ==f'{chr(8738)}':
                mats.append(self.angle_sign())
            elif char == ',':
                one = self.char_mat('1')
                mat = np.ones([28,10])
                mat[-7:] = one[::4,8::2]
                mats.append(mat)
            elif char == '.':
                mat = np.ones([28,7])
                mat[23:26,2:5] = 0 +np.random.rand(3,3)*0.2
                mat = create_handwritten_dot()
                mats.append(1-mat)#[:,8:-8])

            elif char == '°':
                mat = self.char_mat('0')
                downscale_factor = 0.5  # Make it half the size
                new_size = int(28 * downscale_factor)
                mat_small = cv2.resize(mat, (new_size, new_size), interpolation=cv2.INTER_AREA)
                canvas = np.ones((28, 14))
                y_start = 2  # Position at the top
                x_start = 0#28 - new_size - 2  # Position at the right
                canvas[y_start:y_start+new_size] = mat_small
                mats.append(canvas)
            else:
                mats.append(self.char_mat(char)[:,2:-2])
        return np.concatenate(mats,axis=1)
    def text_mat(self,tokens,bbox_width):
        x,y,height = 0,0,28
        max_width = 0 
        token_mats = []
        for token in tokens:
            mat = self.token_mat(token)
            token_mats.append(mat)
            max_width = max(max_width,mat.shape[1])
        ratio = max_width//(bbox_width)+1
        bbox_width *= ratio
        bbox=np.ones([height,bbox_width])
        for mat in token_mats:
            width = mat.shape[1]
            if x+width >= bbox.shape[1]:
                bbox = np.concatenate([bbox,np.ones([height*3//2,bbox_width])],axis = 0)
                x = 0
                y+= height*3//2
            bbox[y:y+height,x:x+width] = mat
            x+= mat.shape[1]+20
        return zoom(bbox,(1/ratio,1/ratio))



def imshow_handwritten(ax,mat,offset_pos,image_size):
    x_min = offset_pos[0] - image_size[0] / 2
    x_max = offset_pos[0] + image_size[0] / 2
    y_min = offset_pos[1] - image_size[1] / 2
    y_max = offset_pos[1] + image_size[1] / 2        
    if (x_max <= x_min) or (y_min >= y_max):
        raise "Image has wrong bbox"
    ax.imshow(mat, cmap='gray', extent=[x_min, x_max, y_min, y_max], zorder=2)  


def find_bbox(ax,padding = 10):
    buf = BytesIO()
    ax.figure.savefig(buf, format='png', bbox_inches='tight', dpi=64)
    buf.seek(0)
    img = Image.open(buf)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_bin = img_cv.mean(axis=2) < 255 ### turn every white into 0, rest are 1    
    best_bbox = [[0,0],[0,0]] ### [x1,x2],[y1,y2]
    best_area = 0
    height,width = img_bin.shape
    corners = [
        (0, 0),          # top-left origin
        (0, width-1),    # top-right origin (flip horizontally)
        (height-1, 0),   # bottom-left origin (flip vertically)
        (height-1, width-1)  # bottom-right origin (flip both)
    ]
    for corner_i, corner_j in corners:
        sub_img = img_bin
        if corner_i == height-1:  #verticle flip 
            sub_img = sub_img[::-1]
        if corner_j == width-1:   # horizontal flip
            sub_img = sub_img[:, ::-1]
        cumsum2d = np.cumsum(np.cumsum(sub_img,axis = 0),axis = 1)
        y_arr,x_arr = np.where(cumsum2d == 0)
        inds = (x_arr > padding)*(x_arr < width-1-padding)*(y_arr > padding)*(y_arr < height-1-padding)*(x_arr <2*y_arr)*(y_arr<2*x_arr)
        
        x_arr = x_arr[inds]
        y_arr = y_arr[inds]
        areas = y_arr*x_arr+x_arr
        try:
            max_area = areas.max()
        except:
            continue
        if best_area<max_area:
            best_area = areas.max()
            ind = areas.argmax()
            x = abs(corner_j - x_arr[ind])
            y = abs(corner_i - y_arr[ind])
            best_bbox = [[min(corner_j,x)+padding,max(corner_j,x)-padding],[min(corner_i,y)+padding,max(corner_i,y)-padding]]
    return best_bbox 


def plot_text_wrapped_bbox(text, bbox, ax=None, max_width_chars=None, **text_kwargs):
    """
    Plot text with automatic wrapping to fit within the bbox width.
    """
    if ax is None:
        ax = plt.gca()
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate approximate characters that can fit horizontally
    if max_width_chars is None:
        # Estimate characters per width (this is approximate)
        avg_char_width = width / 50  # Adjust this heuristic based on your font
        max_width_chars = int(width / avg_char_width)
    
    # Wrap the text
    wrapped_text = textwrap.fill(text, width=max_width_chars)
    
    ax.text(center_x, center_y, wrapped_text, 
            ha='center', va='center',
            **text_kwargs)
    
    return wrapped_text










