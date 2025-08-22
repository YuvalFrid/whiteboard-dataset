import numpy as np 
import string

def Scalene(a,b,rotation = 0,mirror = False,acute = False):
    if mirror:
        b *= -1
    vertices = string.ascii_uppercase   #"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    theta = rotation*np.pi/180

    vertex_perm = [vertices[i] for i in np.random.permutation(range(len(vertices)))[:3]]
    segment_names = [f"{vertex_perm[0]}{vertex_perm[1]}",f"{vertex_perm[0]}{vertex_perm[2]}",
                 f"{vertex_perm[1]}{vertex_perm[2]}",f"{vertex_perm[1]}{vertex_perm[0]}{vertex_perm[2]}",
                 f"{vertex_perm[0]}{vertex_perm[1]}{vertex_perm[2]}",f"{vertex_perm[1]}{vertex_perm[2]}{vertex_perm[0]}"]
    split = np.random.randint(1,4)
    mask_idx = np.random.permutation(np.arange(6))
    mask = np.zeros(6,dtype = bool)
    mask[mask_idx[split:]] = True
    target = segment_names[mask_idx[0]]
    if mask_idx[0] < 3:
        target_type = "find_length"
    else:
        target_type = "find_angle"

    length_ratio = 0.05+np.random.rand(1)[0]*0.4
    if acute:
        length_ratio +=1
    description = {
        "vertices": {
            f"{vertex_perm[0]}": {"x": 0, "y": 0},
            f"{vertex_perm[1]}": {"x": np.cos(theta)*a, "y": np.sin(theta)*a},
            f"{vertex_perm[2]}": {"x": np.cos(theta)*a*length_ratio-np.sin(theta)*b, "y": np.sin(theta)*a*length_ratio+np.cos(theta)*b},
        },
        "segments": {
            segment_names[0]: {"known": mask[0]},
            segment_names[1]: {"known": mask[1]},
            segment_names[2]: {"known": mask[2]},
        },
        "angles": {
            segment_names[3]: {"known": mask[3]},
            segment_names[4]: {"known": mask[4]},
            segment_names[5]: {"known": mask[5]}
        },
        "specials":{},
        "question": {
            "type": target_type,
            "target": f"{target}",
            "question_text": f"Find the size of {target}"
        }
    }
    return description



def Right(a,b,rotation = 0,mirror = False):
    if mirror:
        b *= -1
    vertices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    theta = rotation*np.pi/180

    vertex_perm = [vertices[i] for i in np.random.permutation(range(len(vertices)))[:3]]
    segment_names = [f"{vertex_perm[0]}{vertex_perm[1]}",f"{vertex_perm[0]}{vertex_perm[2]}",
                 f"{vertex_perm[1]}{vertex_perm[2]}",f"{vertex_perm[1]}{vertex_perm[0]}{vertex_perm[2]}",
                 f"{vertex_perm[0]}{vertex_perm[1]}{vertex_perm[2]}",f"{vertex_perm[1]}{vertex_perm[2]}{vertex_perm[0]}"]
    split = np.random.randint(1,4)
    mask_idx = np.random.permutation(np.arange(6))
    mask = np.zeros(6,dtype = bool)
    mask[mask_idx[split:]] = True
    target = segment_names[mask_idx[0]]
    if mask_idx[0] < 3:
        target_type = "find_length"
    else:
        target_type = "find_angle"

    description = {
        "vertices": {
            f"{vertex_perm[0]}": {"x": 0, "y": 0},
            f"{vertex_perm[1]}": {"x": np.cos(theta)*a, "y": np.sin(theta)*a},
            f"{vertex_perm[2]}": {"x": np.cos(theta)*a-np.sin(theta)*b, "y": np.sin(theta)*a+np.cos(theta)*b},
        },
        "segments": {
            segment_names[0]: {"known": mask[0]},
            segment_names[1]: {"known": mask[1]},
            segment_names[2]: {"known": mask[2]},
        },
        "angles": {
            segment_names[3]: {"known": mask[3]},
            segment_names[4]: {"known": mask[4]},
            segment_names[5]: {"known": mask[5]}
        },
        "specials":{},
        "question": {
            "type": target_type,
            "target": f"{target}",
            "question_text": f"Find the size of {target}"
        }
    }
    return description

