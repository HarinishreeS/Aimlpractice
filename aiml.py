from collections import deque
graph={
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

def bfs(start,goal):
    visited=set()
    queue=deque([[start]])

    while queue:
        path=queue.popleft()
        node=path[-1]

        if(node==goal):
            return path
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                new_path=list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None

start_node='A'
goal_node='F'
result=bfs(start_node,goal_node)
print(result)
