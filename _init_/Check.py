def check_char(lake, char):
    lst = []
    for i, state in enumerate(lake):
        temp = []
        for j, c in enumerate(lake):
            if c == char:
                temp.append(1)
            else:
                temp.append(0)
        
        lst.append(temp)
        
    return lst