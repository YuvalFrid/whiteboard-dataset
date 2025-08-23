import numpy as np 
import string
from pdb import set_trace as st 

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

        self.description["vertices"].append({"mark":self.letters[0],"x":0,"y":0}) ### first point always at 0,0
        self.description["vertices"].append({"mark":self.letters[1],"x":a,"y":0}) ### second point always at a,0

        ### Add the first three segments and angles from the triangle limbs
        self.description["segments"].append({"mark":self.letters[0]+self.letters[1]})
        self.description["segments"].append({"mark":self.letters[0]+self.letters[2]})
        self.description["segments"].append({"mark":self.letters[1]+self.letters[2]})
        self.description["angles"].append({"mark":self.letters[0]+self.letters[1]+self.letters[2]})
        self.description["angles"].append({"mark":self.letters[1]+self.letters[2]+self.letters[0]})
        self.description["angles"].append({"mark":self.letters[2]+self.letters[0]+self.letters[1]})

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
        
    def add_median(self,edge ="A",base="BC"):
        ### add a median from vertex edge to line base
        B,C = self._pt(base[0]),self._pt(base[1]) 
        D = (B+C)/2
        new_vertex_mark = self.letters[len(self.description["vertices"])]
        self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        self.description["specials"].append({"type":"median","start":edge,"end":new_vertex_mark,"base":base})
        self.description["segments"].append({"mark":edge+new_vertex_mark})
        self.description["segments"].append({"mark":base[0]+new_vertex_mark,"known":True})
        self.description["segments"].append({"mark":base[1]+new_vertex_mark})
        self.description["angles"].append({"mark":edge+new_vertex_mark+base[0]})
    def add_bisector(self,angle ="ABC"):
        ### Adds a bisector for angle 
        A, B, C = self._pt(angle[0]),self._pt(angle[1]),self._pt(angle[2])
        AB = A - B
        CB = C - B  
        AC = C - A # From A to C 
        ratio = np.linalg.norm(AB)/np.linalg.norm(CB) ## ratio between the two limbs is the same as the two dissected parts, BA/BC = BD/CD
        D = A + AC*(ratio/1+ratio)
        new_vertex_mark = self.letters[len(self.description["vertices"])]
        self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        self.description["specials"].append({"type":"bisector","start":angle[1],"end":new_vertex_mark,"base":angle[0]+angle[2]})
        for i in range(3):
            self.description["segments"].append({"mark":angle[i]+new_vertex_mark})
        self.description["angles"].append({"mark":angle[:2]+new_vertex_mark,"known":True})

    def add_altitude(self,edge = "A",base="BC"):
        ### Adds an altitude from edge to base
        A, B, C = self._pt(edge),self._pt(base[0]),self._pt(base[1])
        BA = A - B
        BC = C - B
        cos_theta = np.dot(BA,BC)/(np.linalg.norm(BA)*np.linalg.norm(BC))
        D = B+BC/np.linalg.norm(BC)*np.linalg.norm(BA)*cos_theta ### Start from B, go in the direction of BC, cos_theta*BA
        new_vertex_mark = self.letters[len(self.description["vertices"])]
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
        new_vertex_mark = self.letters[len(self.description["vertices"])]
        self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        self.description["segments"].append({"mark":base[0]+new_vertex_mark,"known":True})
        BA = A - B 
        DE_dir = np.array([-BA[1],BA[0]])/np.linalg.norm(BA) ### perpendicular direction to BA
        BC = C - B
        ABC = np.arccos(np.dot(BA,BC)/(np.linalg.norm(BA)*np.linalg.norm(BC)))
        AC = C - A 
        BAC = np.arccos(np.dot(-BA,AC)/(np.linalg.norm(BA)*np.linalg.norm(AC)))
        DE_size = np.linalg.norm(BA)*0.5*min(np.tan(ABC),np.tan(BAC))
        E = D-DE_size*DE_dir

########### FIX FOR OBTUSE HERE!


        new_vertex_mark_2 = self.letters[len(self.description["vertices"])]
        self.description["vertices"].append({"mark":new_vertex_mark_2,"x":E[0],"y":E[1]})
        self.description["segments"].append({"mark":new_vertex_mark+new_vertex_mark_2})
        self.description["angles"].append({"mark":base[0]+new_vertex_mark+new_vertex_mark_2,"known":True})
        self.description["specials"].append({"type":"median_perpendicular","start":new_vertex_mark_2,"end":new_vertex_mark,"base":base})


    def add_parallel(self,base = "AB",third="C",ratio = 0.5):
        ### Adds a parallel line to base with ratio
        A, B, C = self._pt(base[0]),self._pt(base[1]),self._pt(third)
        AC = C-A
        BC = C-B 
        new_vertex_mark = self.letters[len(self.description["vertices"])]
        D = A + AC*ratio
        self.description["vertices"].append({"mark":new_vertex_mark,"x":D[0],"y":D[1]})
        new_vertex_mark_2 = self.letters[len(self.description["vertices"])]
        E = B + BC*ratio
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

    

