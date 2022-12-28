if state in self.corners and self.flag == 0:
            print("state in corner")
            self.flag = 2
        
        elif state in self.boundary and self.flag == 0:
            print("state in border")
            self.flag = 1
        
        
        if next_state == state and state in self.corners:
            prob += self.slip_val
            print("p for corners added")
            
        elif next_state == state and state in self.boundary:
            prob += self.slip_val
            print("p for boundaries added")
        
        
        if next_state == state and self.flag != 0:
            print("state prob in border or corner")
            prob += (1 - self.slip)          
        
        
        if state in self.boundary_top and self.flag != 0:
            w = 0
            self.flag -=1
        
        else: 
            if action == 0:
                prob += (1 - self.slip)
        
        if state in self.boundary_left:
            a = 0
            self.flag -=1
        
        else: 
            if action == 1:
                prob += (1 - self.slip)
        
        if state in self.boundary_bottom:
            s = 0
            self.flag -=1
        
        else: 
            if action == 2:
                prob += (1 - self.slip)
            
        if state in self.boundary_right:
            d = 0
            self.flag -=1
        
        else: 
            if action == 3:
                prob += (1 - self.slip)
            
        if next_state in [w, a, s, d]:
            prob += self.slip_val