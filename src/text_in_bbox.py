from pdb import set_trace as st 
import matplotlib.pyplot as plt 

def plot_text_in_bbox(text,bbox):
    x1,y1,x2,y2 = bbox
    height = abs(y2-y1)
    width = abs(x2-x1)
    center = [(x1+x2)/2,(y1+y2)/2]
    ratio = width/height 

    total_text_length = len(text)
    tokens = text.split(" ")
    r_tokens = []
    shift = 0
    print(len(tokens))
    for i in range(len(tokens)-1):
        i+= shift
        if i > len(tokens)-2:
            continue
        if tokens[i+1] == '=':
            r_tokens.append(tokens[i]+" = "+tokens[i+2])
            shift+=2 
        else:
            r_tokens.append(tokens[i])
    
    check_points = []
    check_points.append(len(r_tokens[0])+1)
    for i in range(1,len(r_tokens)):
        check_points.append(len(r_tokens[i])+1+check_points[i-1])
    total_text_squares = total_text_length//2 ### assuming 3 chars are as tall as they are wide on average
    total_tokens_in_row = total_text_squares*ratio ### this ensures a proper ratio between height and width
    rows = []
    start_ind = 0
    total_rows = 1
    for i,token_sum in enumerate(check_points):
        if token_sum > total_rows*total_tokens_in_row:
            if i == len(check_points)-1:
                i+=1 
            rows.append(" ".join(r_tokens[start_ind:i]))
            start_ind = i
            total_rows+=1
    print("TEXT IS ") 
    [print(i) for i in rows]
    st()
    



def tokenize_with_equations(text):
    """
    Tokenize text while preserving equations around '=' signs.
    """
    tokens = text.split(" ")
    r_tokens = []
    i = 0
    
    while i < len(tokens):
        # Check if next token is '=' (and we're not at the end)
        if i < len(tokens) - 2 and tokens[i + 1] == '=':
            # Combine: current + '=' + next
            equation = tokens[i] + " = " + tokens[i + 2]
            r_tokens.append(equation)
            i += 3  # Skip the next two tokens since we've consumed them
        else:
            r_tokens.append(tokens[i])
            i += 1
    
    return r_tokens


if __name__ == "__main__":
    text = "Given AD = 5 and BC = 12 prove CD = 12"
    plot_text_in_bbox(text,[0,0,10,20])
