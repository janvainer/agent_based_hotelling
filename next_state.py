# calculates next state based on agents' actions

def next_state(state,ac1,ac2):
    # arena is [0,..,6]
    # state = (a,b), where a and b are distances from leftmost and rightmost edges
    # a and b \in [0,4]!! 
    # actions are accessed with index!
    
    actions = [-1,0,1]
    st = list(state)
    
    if 0<=state[0]+actions[ac1]<=6:
        st[0] = st[0]+actions[ac1]
    if 0<=state[1]+actions[ac2]<=6:
        st[1] = st[1]+actions[ac2]
        
    return tuple(st)
