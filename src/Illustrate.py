import matplotlib.pyplot as plt 
from src.Aux import * 



def plot_vertices(description,ax,EMNIST,handwritten = False):
    labels = []
    for point in description["vertices"]:
        # Use smart offset for vertex labels
        offset_pos = smart_label_offset(point["x"], point["y"], 
                                      description["vertices"], offset_distance=0.4)
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
        start_ind,end_ind = description["index_lookup"][start],description["index_lookup"][end]
        xs = np.array([description["vertices"][start_ind]["x"], 
                      description["vertices"][end_ind]["x"]])
        ys = np.array([description["vertices"][start_ind]["y"], 
                      description["vertices"][end_ind]["y"]])
          


        if handwritten:
            x_sim,y_sim = simulate_handwritten_line([xs[0],ys[0]],[xs[1],ys[1]],jitter_strength = 0.05*np.random.rand())
            ax.plot(x_sim, y_sim, color='black', linewidth=2, zorder=1)
        else:
            ax.plot(xs, ys, color='black', linewidth=2, zorder=1)
        if s["known"]:
            length = np.sqrt(np.diff(xs)[0]**2 + np.diff(ys)[0]**2)
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
        start, vertex, end = a["mark"]
        if not a["known"]:
            continue
        start_ind,vertex_ind,end_ind = description["index_lookup"][start],description["index_lookup"][vertex],description["index_lookup"][end] 
        A = np.array([description["vertices"][start_ind]["x"], 
                           description["vertices"][start_ind]["y"]], dtype=np.float64)
        B = np.array([description["vertices"][vertex_ind]["x"], 
                           description["vertices"][vertex_ind]["y"]], dtype=np.float64)
        C = np.array([description["vertices"][end_ind]["x"], 
                           description["vertices"][end_ind]["y"]], dtype=np.float64)
        
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
