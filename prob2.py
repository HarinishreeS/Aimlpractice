import heapq
graph = {
    'A':[('B',3),('C',7)],
    'B':[('D',2),('E',5)],
    'C':[('F',1)],
    'D':[],
    'E':[('F',6)],
    'F':[]
}

heuristic = {
    'A':1,
    'B':4,
    'C':9,
    'D':12,
    'E':8,
    'F':0
}
def a_star(start,goal):
    open_list = []
    heapq.heappush(open_list,(heuristic[start],0,[start]))

    while open_list:
        f,g,path=heapq.heappop(open_list)
        node=path[-1]

        if(node==goal):
            return path
    
        for neighbor,cost in graph[node]:
            new_g=g+cost
            new_f=new_g+ heuristic[neighbor]
            new_path=path + [neighbor]
            heapq.heappush(open_list,(new_f,new_g,new_path))

    return None

print("a*=",a_star('A','F'))