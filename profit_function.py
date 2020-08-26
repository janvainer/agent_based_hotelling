def profit(state,p1,p2):
    """Return respective profits
    """    
    
    # scale onto (0,1) interval
    p1 = p1/4
    p2 = p2/4
    s1 = state[0]/7
    s2 = state[1]/7

    T = [0,0]
    
    # If the agents are on the same place:
    if (s2-s1)==0:        
        if p2>p1:
            T[0] = p1
        elif p2<p1:
            T[1] = p2
        else:
            T = [p1/2,p2/2]
    else:
        # Where the cost functions of the marginal customer meet:
        d = 1/2*(p2-p1)/(s2-s1)+(s2+s1)/2    

        # Demands for left and right producer
        Dr = max(0,1-d)
        Dl = max(0,d)
        
        if s1<s2:
            T = [Dl*p1,Dr*p2]
        else:
            T = [Dr*p1,Dl*p2]
    
    return T
    
