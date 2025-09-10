from typing import Dict, List, Any
import numpy as np 
import string
from pdb import set_trace as st 
from typing import List, Union


class Triangle():
    def __init__(self,a,b,rotation = 0,mirror = False):
        self.a = a ### the two parameters needed for our three vertices
        self.b = b*(1-2*mirror) ### if mirror, its minus b, otherwise stays b, for data augmentation 
        self.theta = np.pi*rotation/180 ### rotation, for data augmentation

        self.description = {} ### init json
        features = ["vertices","segments","angles","specials","questions"] ### All lists in description
        for feature in features:
            self.description[feature] = []

        self.letters = string.ascii_uppercase
        if np.random.rand() > 0.8: #### add 20% of the cases with total random letters
            self.letters = [string.ascii_uppercase[i] for i in np.random.permutation(range(len(self.letters)))]
            self.letters = ''.join(self.letters)
        self.description["vertices"].append({"mark":self.letters[0],"x":0,"y":0}) ### first point always at 0,0
        self.description["vertices"].append({"mark":self.letters[1],"x":a,"y":0}) ### second point always at a,0

        ### Add the first three segments and angles from the triangle limbs
        self.description["segments"].append({"mark":self.letters[0]+self.letters[1]})
        self.description["segments"].append({"mark":self.letters[0]+self.letters[2]})
        self.description["segments"].append({"mark":self.letters[1]+self.letters[2]})
        self.description["angles"].append({"mark":self.letters[0]+self.letters[1]+self.letters[2]})
        self.description["angles"].append({"mark":self.letters[1]+self.letters[2]+self.letters[0]})
        self.description["angles"].append({"mark":self.letters[2]+self.letters[0]+self.letters[1]})

    def new_vertex(self,coords):
        threshold = 0.000001
        for v in self.description["vertices"]:
            distance = np.linalg.norm(np.array(coords)-np.array([v["x"],v["y"]]))
            if distance < threshold:
                return v["mark"], False
        return self.letters[len(self.description["vertices"])], True
    def _pt(self, ch):
        ### returns x and y coordinates for vertice ch
        i = self.letters.find(ch)
        v = self.description["vertices"][i]
        return np.array([v["x"], v["y"]], dtype=float)

    def third_vertice(self,triangle_type = "scalene"):
        #### Set the third vertice of the triangle, based on its type
        a,b = self.a,self.b
        if triangle_type == "scalene":
            factor = np.random.rand()*0.4+0.05
            factor *=np.random.choice([-1,1]) ### randomly acute or obtuse
            self.description["vertices"].append({"mark":self.letters[2],"x":factor*a,"y":b}) ### gives height b, and some a that makes sure it isn't isn't isoceles
        elif triangle_type =="right":
            self.description["vertices"].append({"mark":self.letters[2],"x":0,"y":b}) ### makes sure its 90 degrees
        elif triangle_type =="isoceles":
            self.description["vertices"].append({"mark":self.letters[2],"x":0.5*a,"y":b}) 
        elif triangle_type =="equilateral":
            self.description["vertices"].append({"mark":self.letters[2],"x":0.5*a,"y":a*np.sin(np.pi/3)})
        else:
            raise "Not a type from {scalene,right,isoceles, or equilateral}"

    def find_cross(self,A,AD_dir,B,BC_dir):
        ### input is point A and direction vector AD, and point B and direction vector BC. 
        ### output is the point where AD and BC intersect.
        AD_dir = AD_dir/np.linalg.norm(AD_dir) ### assert normal
        BC_dir = BC_dir/np.linalg.norm(BC_dir)
        denom = (AD_dir[0]*BC_dir[1]-AD_dir[1]*BC_dir[0])
        if denom == 0:
            return np.inf
        factor = (AD_dir[0]*(A[1]-B[1])+AD_dir[1]*(B[0]-A[0]))/denom
        return B+BC_dir*factor

    def add_median(self,edge ="A",base="BC"):
        ### add a median from vertex edge to line base
        B,C = self._pt(base[0]),self._pt(base[1]) 
        D = (B+C)/2
        new_vertex_mark,new = self.new_vertex(D)
        if new:
            self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        self.description["specials"].append({"type":"median","start":edge,"end":new_vertex_mark,"base":base})
        self.description["segments"].append({"mark":edge+new_vertex_mark})
        self.description["segments"].append({"mark":base[0]+new_vertex_mark})
        self.description["segments"].append({"mark":base[1]+new_vertex_mark})
        self.description["angles"].append({"mark":edge+new_vertex_mark+base[0]})
    def add_bisector(self,angle ="ABC"):
        ### Adds a bisector for angle 
        A, B, C = self._pt(angle[0]),self._pt(angle[1]),self._pt(angle[2])
        BA = A - B
        BC = C - B  
        AC = C - A # From A to C 
        BD_dir = 0.5*(BA/np.linalg.norm(BA)+BC/np.linalg.norm(BC)) ### directional of bisector
        D = self.find_cross(B,BD_dir,A,AC)
        new_vertex_mark,new = self.new_vertex(D)
        if new:
            self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        self.description["specials"].append({"type":"bisector","start":angle[1],"end":new_vertex_mark,"base":angle[0]+angle[2]})
        for i in range(3):
            self.description["segments"].append({"mark":angle[i]+new_vertex_mark})
        self.description["angles"].append({"mark":angle[:2]+new_vertex_mark})

    def add_altitude(self,edge = "A",base="BC"):
        ### Adds an altitude from edge to base
        A, B, C = self._pt(edge),self._pt(base[0]),self._pt(base[1])
        BA = A - B
        BC = C - B
        cos_theta = np.dot(BA,BC)/(np.linalg.norm(BA)*np.linalg.norm(BC))
        D = B+BC/np.linalg.norm(BC)*np.linalg.norm(BA)*cos_theta ### Start from B, go in the direction of BC, cos_theta*BA
        new_vertex_mark,new = self.new_vertex(D)
        if new:
            self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        self.description["specials"].append({"type":"altitude","start":edge,"end":new_vertex_mark,"base":base})
        self.description["segments"].append({"mark":edge+new_vertex_mark})
        for i in range(2):
            self.description["segments"].append({"mark":base[i]+new_vertex_mark})
        self.description["angles"].append({"mark":edge+new_vertex_mark+base[0],"known":True})


    def add_perpendicular(self,base="AB",third="C"):
        ### Adds a perpendicular median to base 
        A, B, C = self._pt(base[0]),self._pt(base[1]),self._pt(third)
        D = (A+B)/2
        new_vertex_mark,new = self.new_vertex(D)
        if new:
            self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        self.description["segments"].append({"mark":base[0]+new_vertex_mark,"known":True})
        BA = A - B 
        DE_dir = np.array([-BA[1],BA[0]])/np.linalg.norm(BA) ### perpendicular direction to BA
        BC = C - B
        AC = C - A
        E1 = self.find_cross(D,DE_dir,B,BC)
        E2 = self.find_cross(D,DE_dir,A,AC)
        if np.linalg.norm(D-E1) > np.linalg.norm(D-E2):
            E = E2
        else:
            E = E1
        new_vertex_mark_2, new = self.new_vertex(E)
        if new:
            self.description["vertices"].append({"mark":new_vertex_mark_2,"x":E[0],"y":E[1]})
        self.description["segments"].append({"mark":new_vertex_mark+new_vertex_mark_2})
        self.description["angles"].append({"mark":base[0]+new_vertex_mark+new_vertex_mark_2,"known":True})
        self.description["specials"].append({"type":"median_perpendicular","start":new_vertex_mark_2,"end":new_vertex_mark,"base":base})


    def add_parallel(self,base = "AB",third="C",ratio = 0.5):
        ### Adds a parallel line to base with ratio
        A, B, C = self._pt(base[0]),self._pt(base[1]),self._pt(third)
        AC = C-A
        BC = C-B 
        D = A + AC*ratio
        new_vertex_mark,new = self.new_vertex(D)
        if new:
            self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        E = B + BC*ratio
        new_vertex_mark_2,new = self.new_vertex(E)
        if new:
            self.description["vertices"].append({"mark":new_vertex_mark_2,"x":E[0],"y":E[1]})        
        self.description["segments"].append({"mark":new_vertex_mark+new_vertex_mark_2})
        self.description["specials"].append({"type":"parallel_line","mark":new_vertex_mark+new_vertex_mark_2,"base":base})

    def set_question(self):
        ### sets all known and unknowns, and the questions
        num_segments = len(self.description["segments"])
        num_angles = len(self.description["angles"])
        mask = np.zeros(num_segments+num_angles,dtype = bool)
        mask_idx = np.random.permutation(range(mask.size)) ### random order 
        split = np.random.randint(mask.size//2-2,mask.size//2+2) ## everything before split is unknown, after is known. 
        mask[mask_idx[split:]] = True 
        for i in range(num_segments):
            self.description["segments"][i]["known"] = mask[i]
        for i in range(num_angles):
            self.description["angles"][i]["known"] = mask[i+num_segments]
        questions = np.random.choice(range(1,min(4,split+1)))
        for q in range(questions):
            ind = mask_idx[q]
            if ind < num_segments:
                target_type = "Find segment "
                target = self.description["segments"][ind]["mark"]
            else:
                target_type = "Find angle "
                target = self.description["angles"][ind - num_segments]["mark"]
            self.description["questions"].append(target_type + target)

    def rotate(self):
        ### Should be done at the end after setting everything up
        for vertix in self.description["vertices"]:
            x_rot = np.cos(self.theta)*vertix["x"]-np.sin(self.theta)*vertix["y"]
            y_rot = np.cos(self.theta)*vertix["y"]+np.sin(self.theta)*vertix["x"]
            vertix["x"],vertix["y"] = x_rot,y_rot

    
    def set_3V_question(self,num_questions = 1):
        #### all questions with sine or cosine theorems. only 3 values are needed in total - 1-3 segments, 0-2 angles. 
        mask = np.zeros(6,dtype = bool)
        mask[:3] = True ### only three values in total are needed
        mask = np.random.permutation(mask) ### random permutation
        if mask[3:].sum() == 3: ### in case all angles are given and no limbs.
            mask[np.random.choice(range(3))] = True
            mask[np.random.choice(range(3,6))] = False
        for i in range(3):
            self.description["segments"][i]["known"] = mask[i]
            self.description["angles"][i]["known"] = mask[i+3]
        question_inds = np.where(mask == False)[0]
        for question in range(num_questions):
            i = question_inds[question]
            if i < 3:
                self.description["questions"].append("Find Segment " +self.description["segments"][i]["mark"]+". ")
            else:
                self.description["questions"].append("Find Angle ∢"+self.description["angles"][i-3]["mark"]+". ")

    def set_4V_median_question(self,num_questions = 1):
        vertex_order = np.random.permutation(range(3)) ## randomly set the edge and base
        edge = self.description["vertices"][vertex_order[0]]["mark"]
        base = self.description["vertices"][vertex_order[1]]["mark"]+self.description["vertices"][vertex_order[2]]["mark"]
        base = "".join(sorted(base))
        self.add_median(edge = edge,base =base)
        end = self.description["vertices"][3]["mark"]
        self.description = canonize_geometry_preserve(self.description)

        segment_order = [i["mark"] for i in self.description["segments"]]
        angle_order = [i["mark"] for i in self.description["angles"]]
        segment_order = np.random.permutation(segment_order)
        angle_order = np.random.permutation(angle_order)
        angle_num = np.random.randint(3) ### 0 1 2 angles options
        unknown = np.array(list(angle_order[angle_num:]) + list(segment_order[3-angle_num:]))
        known = np.array(list(angle_order[:angle_num]) + list(segment_order[:3-angle_num]))

        for part in self.description["segments"]+self.description["angles"]:
            if part["mark"] in known:
                part["known"] = True
            else:
                part["known"] = False
        for question in range(num_questions):
            mark = unknown[question]
            if len(mark) == 2:
                self.description["questions"].append(f"Find segment {mark}.")
            else:
                self.description["questions"].append(f"Find angle ∢{mark}.")

    def set_4V_altitude_question(self,num_questions = 1):
        vertex_order = np.random.permutation(range(3)) ## randomly set the edge and base
        A,B,C = [np.array([self.description["vertices"][v]["x"],self.description["vertices"][v]["y"]]) for v in vertex_order]
        if np.dot(A-B,B-C) == 0:  ## hypotenuse AC 
            edge = self.description["vertices"][vertex_order[1]]["mark"]
            base = self.description["vertices"][vertex_order[0]]["mark"]+self.description["vertices"][vertex_order[2]]["mark"]
        elif np.dot(A-C,C-B) == 0: ### hypotenuse AB 
            edge = self.description["vertices"][vertex_order[2]]["mark"]
            base = self.description["vertices"][vertex_order[0]]["mark"]+self.description["vertices"][vertex_order[1]]["mark"]

        else: ### hypotenuse BC or not right angle
            edge = self.description["vertices"][vertex_order[0]]["mark"]
            base = self.description["vertices"][vertex_order[1]]["mark"]+self.description["vertices"][vertex_order[2]]["mark"]
        base = "".join(sorted(base))
        self.add_altitude(edge = edge,base =base)
        end = self.description["vertices"][3]["mark"]
        self.description = canonize_geometry_preserve(self.description)
        segment_order = [i["mark"] for i in self.description["segments"]]
        angle_order = [i["mark"] for i in self.description["angles"]]
        segment_order = np.random.permutation(segment_order)
        angle_order = np.random.permutation(angle_order)
        
        angle_num = np.random.randint(1) ### 1 angles options
        unknown = np.array(list(angle_order[angle_num:]) + list(segment_order[3-angle_num:]))
        known = np.array(list(angle_order[:angle_num]) + list(segment_order[:3-angle_num]))

        for part in self.description["segments"]+self.description["angles"]:
            if part["mark"] in known:
                part["known"] = True
            else:
                part["known"] = False
        for question in range(num_questions):
            mark = unknown[question]
            if len(mark) == 2:
                self.description["questions"].append(f"Find segment {mark}.")
            else:
                self.description["questions"].append(f"Find angle ∢{mark}.")



    def set_4V_bisector_question(self,num_questions = 1):
        vertex_order = np.random.permutation(range(3)) ## randomly set the edge and base
        edge = self.description["vertices"][vertex_order[0]]["mark"]
        base = self.description["vertices"][vertex_order[1]]["mark"]+self.description["vertices"][vertex_order[2]]["mark"]
        base = "".join(sorted(base))
        self.add_bisector(base[0]+edge+base[1])

        end = self.description["vertices"][3]["mark"]
        self.description = canonize_geometry_preserve(self.description)

        segment_order = [i["mark"] for i in self.description["segments"]]
        angle_order = [i["mark"] for i in self.description["angles"]]
        segment_order = np.random.permutation(segment_order)
        angle_order = np.random.permutation(angle_order)
        angle_num = np.random.randint(3) ### 0 1 2 angles options
        unknown = np.array(list(angle_order[angle_num:]) + list(segment_order[3-angle_num:]))
        known = np.array(list(angle_order[:angle_num]) + list(segment_order[:3-angle_num]))
        counter = 0
        for part in self.description["segments"]+self.description["angles"]:
            if part["mark"] in known:
                part["known"] = True
                counter += 1
            else:
                part["known"] = False

        for question in range(num_questions):
            mark = unknown[question]
            if len(mark) == 2:
                self.description["questions"].append(f"Find segment {mark}.")
            else:
                self.description["questions"].append(f"Find angle ∢{mark}.")






def canonize_geometry_preserve(description):
    marks = []
    vertices = []
    for v in description["vertices"]:
        marks.append(v["mark"])
        vertices.append(v)

    inds = sorted(range(len(marks)), key = lambda k:marks[k])
    for i in inds:
        description["vertices"][i] = vertices[inds[i]]

    marks = []
    segments = []
    for s in description["segments"]:
        s["mark"] = "".join(sorted(s["mark"]))
        marks.append(s["mark"])
        segments.append(s)
    inds = sorted(range(len(marks)), key = lambda k:marks[k])
    for i in inds:
        description["segments"][i] = segments[inds[i]]

    marks = []
    angles = []
    for a in description["angles"]:
        base = a["mark"][1]
        edges = sorted([a["mark"][i] for i in [0,2]])        
        a["mark"] = edges[0]+base+edges[1]
        marks.append(a["mark"])
        angles.append(a)

    inds = sorted(range(len(marks)), key = lambda k:marks[k])
    for i in inds:
        description["angles"][i] = angles[inds[i]]

    
    return description

