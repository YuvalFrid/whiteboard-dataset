import matplotlib.pyplot as plt 
from src.Aux import * 
import cv2
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import textwrap
from matplotlib.font_manager import FontProperties
from math import atan2
from scipy.optimize import linprog


def plot_vertices(description,ax,EMNIST,handwritten = False):
    labels = []
    x,y = 0,0
    for point in description["vertices"][:3]:
        x += point["x"]/3
        y += point["y"]/3
    O = np.array([x,y])
    for point in description["vertices"]:
        # Use smart offset for vertex labels
        offset_pos = smart_label_offset(point["x"]+0.05*(point["x"]-O[0]), point["y"]+0.05*(point["y"]-O[1]), 
                                      description["vertices"], offset_distance=0.4)
        vertice = np.array([point["x"],point["y"]])
     #   offset_pos = vertice + (vertice-O)*0.1
        if handwritten:
            mat = EMNIST.char_mat(point["mark"])
            imshow_handwritten(ax,mat,offset_pos,[0.5,0.5])        
        else:
            label = ax.text(offset_pos[0], offset_pos[1], point["mark"], fontsize=14, 
                           fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.8))               
            labels.append(label)

    return labels

def plot_segments(description,ax,EMNIST,handwritten = False):
    segment_labels = []
    lower_letters = string.ascii_lowercase
    for s in description["segments"]:
        start, end = s["mark"]
        inds = [description["index_lookup"][mark] for mark in s["mark"]]

        A, B = [np.array([description["vertices"][ind]["x"],description["vertices"][ind]["y"]],dtype = np.float64) for ind in inds]
        xs,ys = [A[0],B[0]],[A[1],B[1]]  
        if handwritten:
            x_sim,y_sim = simulate_handwritten_line(A,B,jitter_strength = 0.05*np.random.rand())
            ax.plot(x_sim, y_sim, color='black', linewidth=2, zorder=1)
        else:
            ax.plot(xs, ys, color='black', linewidth=2, zorder=1)
        if s["known"]:
            length = np.linalg.norm(B-A)
            length = length#*(0.7+0.6*np.random.rand())### add +-30% for randomization
            real_value_disc = True#bool(random.getrandbits(1))
            formatted_length, is_sqrt = format_length_with_sqrt(length)
        
            if real_value_disc:
                length_val = np.around(length,2)    
            else:
                length_val = lower_letters[np.random.choice(range(len(lower_letters)))]
                formatted_length = length_val
            if s["known"]:
                s["length"] = length_val
                s["unit"] = "cm"
                
            continue
            # Calculate perpendicular offset for segment labels
            segment_vector = np.array([np.diff(xs)[0], np.diff(ys)[0]])
            segment_length = np.linalg.norm(segment_vector)
            perpendicular = np.array([-segment_vector[1], segment_vector[0]]) / segment_length
            
            # Offset the label perpendicular to the segment
            offset_distance = 0.2
            label_x = xs.mean() + perpendicular[0] * offset_distance
            label_y = ys.mean() + perpendicular[1] * offset_distance
            
            bg_color = 'white'
            if handwritten:
                if type(length_val) == str:
                    mat = EMNIST.char_mat(length_val)
                    width = 0.5
                else:
                    mat =   EMNIST.float_mat(f'{formatted_length}')
                    width = 0.5*(1+len(formatted_length)//2)                
                imshow_handwritten(ax,mat,[label_x,label_y],[width,0.5])                                 

            else:
                label = ax.text(label_x, label_y, f'{formatted_length}', fontsize=12,
#                label = ax.text(label_x, label_y, f'{s}={formatted_length}', fontsize=12,
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color, 
                                edgecolor='none', alpha=0.8))
                segment_labels.append(label)
    return description,segment_labels


def plot_angles(description,ax,EMNIST,handwritten = False):
    angle_labels = []
    for a in description["angles"]:
        if not a["known"]:
            continue
        angle_inds = [description["index_lookup"][i] for i in a["mark"]]
        A,B,C = [np.array([description["vertices"][ind]["x"],description["vertices"][ind]["y"]], dtype=np.float64) for ind in angle_inds]
        
        BA = A-B 
        BC = C-B
        BA_size = np.linalg.norm(BA)
        BC_size = np.linalg.norm(BC)
        degree = np.around(np.arccos(np.dot(BA,BC)/(BA_size*BC_size))*180/np.pi,2)
        if not degree == 90:
            a["value"] = np.around(degree,2)#*(0.9+0.2*np.random.rand()),2) ### randomize with +-10 % 
        else:
            a["value"] = 90
        a["unit"] = "deg"
        radius = 0.2 * min(BA_size, BC_size)
        
        BA_dir = BA/BA_size
        BC_dir = BC/BC_size
        
        if degree == 90:
            # Draw right angle marker
            corner_size = radius * 0.7
            corner1 = B + BA_dir * corner_size
            corner2 = B + BC_dir * corner_size
            corner3 = corner1 + BC_dir * corner_size
             
            ax.plot([corner1[0], corner3[0]], [corner1[1], corner3[1]], 
                   'black', linewidth=2)
            ax.plot([corner2[0], corner3[0]], [corner2[1], corner3[1]], 
                   'black', linewidth=2)
        else:
            draw_handwritten_arc(B + BA_dir * radius, B, 
                            B + BC_dir * radius,jitter_amount = 0.1*np.random.rand()*handwritten)
            # Position angle label
            vec_mid = BA+BC
            vec_mid_normalized = vec_mid/np.linalg.norm(vec_mid)
            label_radius = radius * 0.7  # Place label inside the arc
            
            label_pos = B + vec_mid_normalized * label_radius
            continue
            if handwritten:
                mat = EMNIST.float_mat(f'{degree}°')
                imshow_handwritten(ax,mat,label_pos,[1,0.5])
            else:
                label = ax.text(label_pos[0], label_pos[1], f'{degree}°', fontsize=12,
                           ha='center', va='center', color='black', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                    edgecolor='none', alpha=0.8))
                angle_labels.append(label)
    return description, angle_labels


def mark_line(description,mark = "AB",num = 1):
    ### receives a segment to be marked, returns a list of locations for marking.
    inds = [description["index_lookup"][mark[i]] for i in range(2)]
    A,B = [np.array([description["vertices"][ind]["x"],description["vertices"][ind]["y"]]) for ind in inds]
    AB = B-A
    AB_dir = AB/np.linalg.norm(AB)
    AB_per = np.array([-AB[1],AB[0]])
    AB_per = 0.5*AB_per/np.linalg.norm(AB_per)
    center = A+AB*(np.random.rand()*0.1+0.2) ### place it somewhere between 40%-60% of the segment, close to the middle.
    locs = []
    for i in range(num):
        shift = i*(num-1)/2
        mark = center - shift*0.1*AB +num/2*0.1*AB
        locs.append([mark-AB_per,mark+AB_per]) 
    return locs



def mark_angle(description,mark = "ABC",num = 1):
    ### receives a segment to be marked, returns a list of locations for marking.
    inds = [description["index_lookup"][mark[i]] for i in range(3)]
    A,B,C = [np.array([description["vertices"][ind]["x"],description["vertices"][ind]["y"]]) for ind in inds]
    BA = A - B 
    BC = C - B
    BA_size = np.linalg.norm(BA)
    BC_size = np.linalg.norm(BC)
    radius = 0.2 * min(BA_size, BC_size)
    bisector = 0.5*(BA/BA_size+BC/BC_size)
    bisector_per = 0.5*np.array([-bisector[1],bisector[0]])
    center = B+radius*bisector
    locs = []
    for i in range(num):
        shift = i*(num-1)/2
        mark = center - shift*0.3*bisector_per + num/2*0.1*bisector_per
        locs.append([mark-0.5*bisector,mark+0.5*bisector])
    return locs

def mark_parallel(description,mark1 = "AB",mark2 = "CD"):
    inds = [description["index_lookup"][mark1[i]] for i in range(2)]
    A,B = [np.array([description["vertices"][ind]["x"],description["vertices"][ind]["y"]]) for ind in inds]
    AB = B-A
    inds = [description["index_lookup"][mark2[i]] for i in range(2)]
    C,D = [np.array([description["vertices"][ind]["x"],description["vertices"][ind]["y"]]) for ind in inds]
    CD = D-C
    if np.dot(AB,CD) < 0: ### flipped directions
        CD = -CD
    AB_dir = AB/np.linalg.norm(AB)
    AB_per = np.array([-AB[1],AB[0]])
    AB_per = AB_per/np.linalg.norm(AB_per)
    center = A+AB*(0.1+np.random.rand()*0.2)
    arrow_limb = AB_dir*(1+np.random.rand()) + AB_per
    arrow_size = 2*np.linalg.norm(arrow_limb)
    locs =[]
    for vertex,vector in zip([A,C],[AB,CD]):
        center = vertex+vector*(0.1+np.random.rand()*0.2)
        locs.append([center,center+arrow_limb/arrow_size])
        locs.append([center,center+(arrow_limb-2*AB_per)/arrow_size])
    return locs



def plot_specials(description,ax,EMNIST,handwritten = False):
    line_marks = 0
    angle_marks = 0
    for s in description["specials"]:
        if s['type'] == 'median' or s['type'] == 'median_perpendicular':
        #### Plots two equal marks for median or perpendicular
            line_marks += 1
            for edge in s["base"]:
                locs = mark_line(description,mark = s["end"]+edge,num = line_marks)
                for loc in locs:
                    if handwritten:
                        x_sim,y_sim = simulate_handwritten_line(loc[0],loc[1],jitter_strength = 0.05*np.random.rand())
                        ax.plot(x_sim, y_sim, color='black', linewidth=2, zorder=1)
                    else:
                        ax.plot([loc[0][0],loc[1][0]], [loc[0][1],loc[1][1]], color='black', linewidth=2, zorder=1)
        if s['type'] == 'altitude' or s['type'] == 'median_perpendicular':
            angle_mark = min(s['start'],s['base'][0])+s['end']+max(s['start'],s['base'][0])
            exist = False
            for angle in description["angles"]:
                if angle_mark == angle["mark"] or angle_mark[::-1] == angle["mark"]:
                    angle["known"] = True
                    exist = True
            if not exist:
                description["angles"].append({"mark":angle_mark,"known":True})
        if s['type'] == 'bisector':
            angle_marks += 1 
            head_angle = s['base'][0]+s['start']+s['base'][1]
            for angle in description["angles"]:
                if head_angle == angle["mark"] or head_angle[::-1] == angle["mark"]:
                    angle["known"] = False
            for start in s['base']:
                angle_mark = start+s['start']+s['end']
                exist = False
                head_angle_exist = False
                for angle in description["angles"]:
                    if angle_mark == angle["mark"] or angle_mark[::-1] == angle["mark"]:
                        angle["known"] = True
                        exist = True
                if not exist:
                    description["angles"].append({"mark":angle_mark,"known":True})
                locs = mark_angle(description,angle_mark,num = angle_marks)
                for loc in locs:
                    if handwritten:
                        x_sim,y_sim = simulate_handwritten_line(loc[0],loc[1],jitter_strength = 0.05*np.random.rand())
                        ax.plot(x_sim, y_sim, color='black', linewidth=2, zorder=1)
                    else:
                        ax.plot([loc[0][0],loc[1][0]], [loc[0][1],loc[1][1]], color='black', linewidth=2, zorder=1)
        if s['type'] =='parallel_line':
            mark,base = s['mark'],s['base']            
            locs = mark_parallel(description,mark,base)
            for loc in locs:
                if handwritten:
                    x_sim,y_sim = simulate_handwritten_line(loc[0],loc[1],jitter_strength = 0.05*np.random.rand())
                    ax.plot(x_sim, y_sim, color='black', linewidth=2, zorder=1)
                else:
                    ax.plot([loc[0][0],loc[1][0]], [loc[0][1],loc[1][1]], color='black', linewidth=2, zorder=1)

    return description






def tokenize_with_equations(text):
    """
    Tokenize text while preserving equations around '=' signs.
    """
    tokens = text.split(" ")
    r_tokens = []
    i = 0
    special_symbols = {"=", "∥", "⊥", "~", "≅"}

    while i < len(tokens):
        # Check if next token is '=' (and we're not at the end)
        if i < len(tokens) - 2 and tokens[i + 1] in special_symbols:
            # Combine: current + '=' + next
            equation = tokens[i] + " = " + tokens[i + 2]
            r_tokens.append(equation)
            i += 3  # Skip the next two tokens since we've consumed them
        else:
            r_tokens.append(tokens[i])
            i += 1

    return r_tokens


def render_wrapped_text(tokens, width, height, fontsize=12, fontname="DejaVu Sans", min_fontsize=6):
    """
    Renders a list of tokens (words) into a bounding box of given width and height.
    Automatically reduces font size if text doesn't fit.

    Args:
        tokens: List of words (strings) to render.
        width: Width of the bounding box.
        height: Height of the bounding box.
        fontsize: Initial font size for the text.
        fontname: Font family for the text.
        min_fontsize: Minimum font size to try.

    Returns:
        numpy.ndarray: Rendered image as a numpy array (RGB format).
    """
    try:
        return _render_with_fontsize(tokens, width, height, fontsize, fontname)
    except TextDoesNotFitError:
        if fontsize > min_fontsize:
            # Recursively try with smaller font size
            return render_wrapped_text(tokens, width, height, fontsize - 1, fontname, min_fontsize)
        else:
            # Use minimum font size and render what fits
            return _render_with_fontsize(tokens, width, height, min_fontsize, fontname, force_render=True)


def _render_with_fontsize(tokens, width, height, fontsize, fontname, force_render=False):
    """
    Internal function to render with specific font size.
    Raises TextDoesNotFitError if text doesn't fit and force_render is False.
    """
    # Create a figure and axis with tight layout
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Remove all margins
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')

    # Initialize position
    x, y = 0, height  # Start from top-left
    font_prop = FontProperties(family=fontname, size=fontsize)

    # Place each token
    for token in tokens:
        # Draw temporarily to measure
        t = ax.text(x, y, token, fontproperties=font_prop, ha='left', va='top', alpha=0)
        renderer = ax.figure.canvas.get_renderer()
        bb = t.get_window_extent(renderer=renderer)
        bb_data = ax.transData.inverted().transform(bb.corners())
        tok_w = bb_data[:, 0].max() - bb_data[:, 0].min()
        tok_h = bb_data[:, 1].max() - bb_data[:, 1].min()
        t.remove()

        # Check if the token fits in the current line
        if x + tok_w > width:
            # Move to the next line
            x = 0
            y -= 1.2 * tok_h  # Move down for the next line

        # Check if the token fits vertically
        if y - tok_h < 0:
            if force_render:
                # In force mode, just skip this token
                continue
            else:
                plt.close(fig)
                raise TextDoesNotFitError(f"Text doesn't fit with fontsize {fontsize}")

        # Place the token
        ax.text(x, y, token, fontproperties=font_prop, ha='left', va='top')

        # Update x position for the next token
        x += tok_w + fontsize * 0.3  # Add spacing

    # Convert the figure to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


class TextDoesNotFitError(Exception):
    """Exception raised when text doesn't fit in the bounding box."""
    pass





def overlay_text_on_image(base_img, text_img, bbox):
    """
    Overlay text_img on base_img at the bbox location.
    bbox = (x, y, width, height)
    """
    x, y, w, h = bbox

    text_img_resized = Image.fromarray(text_img).resize((w, h))
    text_img_resized = np.array(text_img_resized)

    base_copy = base_img.copy()
    base_copy[y:y+h, x:x+w] = text_img_resized
    return base_copy








def order_ccw(pts):
    """Return pts ordered CCW around centroid."""
    pts = np.asarray(pts, dtype=float)
    cx, cy = pts.mean(axis=0)
    angles = [atan2(y - cy, x - cx) for x, y in pts]
    order = np.argsort(angles)
    return pts[order]

def halfspaces_of_convex_polygon(vertices_ccw):
    """
    Given CCW vertices of a convex polygon, return array of unit outward normals N (m,2)
    and offsets b (m,) s.t. N[i]·x <= b[i] for points inside the polygon.
    """
    V = np.asarray(vertices_ccw, dtype=float)
    m = V.shape[0]
    normals = []
    offsets = []
    for i in range(m):
        p = V[i]
        q = V[(i+1) % m]
        edge = q - p            # edge vector from p->q (polygon CCW)
        # outward normal for CCW polygon: rotate edge clockwise (dy, -dx)
        nvec = np.array([edge[1], -edge[0]], dtype=float)
        norm = np.linalg.norm(nvec)
        if norm == 0:
            continue
        u = nvec / norm         # unit outward normal
        b = np.dot(u, p)        # n·x = b for points on the edge
        normals.append(u)
        offsets.append(b)
    return np.asarray(normals), np.asarray(offsets)

def largest_axis_aligned_rect_in_kite(vertices, R, tol=1e-9):
    """
    vertices: list of 4 (x,y) for kite ABCD in any order
    R: height/width ratio (H/W)
    Returns dict: W, H, center (cx,cy), corners (4x2), success flag.
    """
    # order CCW to be safe
    Vccw = order_ccw(vertices)
    N, b = halfspaces_of_convex_polygon(Vccw)
    if N.size == 0:
        return {"success": False, "reason": "bad polygon"}

    # alpha_i = 0.5*(|n_x| + R*|n_y|)
    alpha = 0.5 * (np.abs(N[:,0]) + R * np.abs(N[:,1]))

    # variables: [cx, cy, W] -> maximize W  <=> minimize -W
    c_obj = np.array([0.0, 0.0, -1.0])

    # inequalities: N @ [cx,cy] + alpha * W <= b
    A_ub = np.hstack([N, alpha.reshape(-1,1)])
    b_ub = b.copy()

    # bounds: cx, cy free; W >= 0
    bounds = [(None, None), (None, None), (0, None)]

    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success or res.x[2] <= tol:
        return {"success": False, "reason": "no feasible rectangle found", "lp_res": res}

    cx, cy, W = res.x
    H = R * W
    dx, dy = W/2.0, H/2.0
    corners = np.array([
        [cx - dx, cy - dy],
        [cx + dx, cy - dy],
        [cx + dx, cy + dy],
        [cx - dx, cy + dy],
    ])

    return {"success": True, "W": W, "H": H, "center": (cx, cy), "corners": corners, "lp_res": res}

