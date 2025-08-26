import matplotlib.pyplot as plt 
from src.Aux import * 
import cv2
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


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
            real_value_disc = bool(random.getrandbits(1))
            formatted_length, is_sqrt = format_length_with_sqrt(length)
        
            if real_value_disc:
                length_val = np.around(length,2)    
            else:
                length_val = lower_letters[np.random.choice(range(len(lower_letters)))]
                formatted_length = length_val
            if s["known"]:
                s["length"] = length_val
                s["unit"] = ["cm"]
                
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
        a["value"] = degree
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
            angle_mark = s['start']+s['end']+s['base'][0]
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

    return description




