{{{
def ucs(graph, start):
    queue = [(0, [start])]  # Initialize the queue with start node and its cost
    visited = set()  # Set to keep track of visited nodes

    while queue:
        cost, path = queue.pop(0)  # Get the path with minimum cost
        node = path[-1]  # Get the last node in the path

        if node == 'G':  # Check if the goal node is reached
            return path, cost  # Return the path and its cost

        if node not in visited:
            visited.add(node)  # Mark the current node as visited

            if node in graph:  # Check if the current node has neighbors
                for neighbor, edge_cost in graph[node].items():
                    new_cost = cost + edge_cost
                    new_path = path + [neighbor]
                    queue.append((new_cost, new_path))  # Add new path to the queue

                queue.sort()  # Sort the queue based on cost

    return None, None  # Return None if no path is found

# Define the graph
graph = {
    'S': {'A': 3, 'C': 2, 'D': 2},
    'D': {'B': 3, 'G': 8},
    'B': {'E': 2},
    'E': {'G': 2},
    'C': {'F': 1},
    'F': {'E': 0, 'G': 4},
}

# Call the function with start node 'S'
path, cost = ucs(graph, 'S')
print("Shortest path:", path)
print("Total cost:", cost)
}}}


























{{{
def astar(start, goal, graph):
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    h_score = {start: heuristic(start)}

    while open_set:
        current = min(open_set, key=lambda x: g_score.get(x, float('inf')) + h_score.get(x, float('inf')))

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        open_set.remove(current)

        for neighbor, cost in graph.get(current, []):
            tentative_g_score = g_score.get(current, float('inf')) + cost
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h_score[neighbor] = heuristic(neighbor)
                open_set.add(neighbor)

    return None

def heuristic(node):
    H_dist = {
        'A': 11,
        'B': 6,
        'C': 99,
        'D': 1,
        'E': 7,
        'G': 0,
    }
    return H_dist[node]

graph = {
    "A": [('B', 2), ('E', 3)],
    "B": [('C', 1), ('G', 9)],
    "C": [],
    "E": [('D', 6)],
    "D": [('G', 1)],
}

path = astar('A', 'G', graph)
print("Path found:", path)
}}}
















{{{
def ida_star(start, goal):
    graph = {
        "A": [('B', 2), ('E', 3)],
        "B": [('C', 1), ('G', 9)],
        "C": [],
        "E": [('D', 6)],
        "D": [('G', 1)],
    }

    def heuristic(node):
        H_dist = {
            'A': 11,
            'B': 6,
            'C': 99,
            'D': 1,
            'E': 7,
            'G': 0,
        }
        return H_dist[node]

    def search(path, g, bound):
        node = path[-1]
        f = g + heuristic(node)
        if f > bound:
            return f
        if node == goal:
            return 'FOUND'
        min_cost = float('inf')
        for neighbor, cost in graph.get(node, []):
            if neighbor not in path:
                path.append(neighbor)
                new_cost = search(path, g + cost, bound)
                if new_cost == 'FOUND':
                    return 'FOUND'
                if new_cost < min_cost:
                    min_cost = new_cost
                path.pop()
        return min_cost

    bound = heuristic(start)
    path = [start]
    while True:
        cost = search(path, 0, bound)
        if cost == 'FOUND':
            return path
        if cost == float('inf'):
            return None
        bound = cost

# Example usage:
path = ida_star('A', 'G')
print("Path found:", path)
}}}







{{{
graph = {
    "S": ['B', 'C'],
    "B": ['D', 'E'],
    "C": ['F'],
    "D": [],
    "E": ['F'],
    "F": []
}

visited = set()

def dfs(graph, node):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbour in graph.get(node, []):
            dfs(graph, neighbour)

dfs(graph, "S")
}}}













{{{
graph={"A":['B','C'],"B":['D','E'],"C":['F'],"D":[],"E":['F'],"F":[]}
visited=[]
queue=[]    
def bfs(visited,graph,node):
    visited.append(node)
    queue.append(node)
    while queue:
        s=queue.pop(0)
        print(s,end="")
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
bfs(visited,graph,'A')
}}}













{{{BFS Shortest path
graph={"A":[('B',12),('C',4)],
       "B":[('D',7),('E',3)],
       "C":[('F',8),('G',2)],
       "D":[],
       "E":[('H',0)],
       "F":[('H',0)],
       "G":[('H',0)]
      }
def bfs(start,target,graph,queue=[],visited=[]):
    if start not in visited:
        print(start)
        visited.append(start)
    queue=queue+[x for x in graph[start] if x[0][0] not in visited]
    queue.sort(key=lambda x:x[1])
    if queue[0][0]==target:
        print(queue[0][0])
    else:
        processing=queue[0]
        queue.remove(processing)
        bfs(processing[0],target,graph,queue,visited)
        
bfs('A','H',graph)
}}}
















{{{best FS
from queue import PriorityQueue

graph = {
    "A": [('B', 5), ('C', 7)],
    "B": [('D', 3), ('E', 2)],
    "C": [('F', 4)],
    "D": [],
    "E": [('F', 1)],
    "F": []
}

def best_first_search(graph, start, goal):
    visited = set()
    queue = PriorityQueue()
    queue.put((0, start))

    while not queue.empty():
        cost, node = queue.get()
        if node == goal:
            return visited
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            for neighbor, weight in graph.get(node, []):
                if neighbor not in visited:
                    queue.put((weight, neighbor))

    return None

# Example usage:
best_first_search(graph, 'A', 'F')
}}}












{{{BST
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if not self.root:
            self.root = Node(key)
        else:
            self._insert_recursively(self.root, key)

    def _insert_recursively(self, node, key):
        if key < node.val:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert_recursively(node.left, key)
        else:
            if node.right is None:
                node.right = Node(key)
            else:
                self._insert_recursively(node.right, key)

    def search(self, key):
        return self._search_recursively(self.root, key)

    def _search_recursively(self, node, key):
        if node is None or node.val == key:
            return node
        if key < node.val:
            return self._search_recursively(node.left, key)
        return self._search_recursively(node.right, key)

    def inorder_traversal(self):
        self._inorder_traversal_recursive(self.root)

    def _inorder_traversal_recursive(self, node):
        if node:
            self._inorder_traversal_recursive(node.left)
            print(node.val, end=" ")
            self._inorder_traversal_recursive(node.right)

# Example usage:
bst = BinarySearchTree()
keys = [50, 30, 70, 20, 40, 60, 80]
for key in keys:
    bst.insert(key)

print("Inorder traversal of the BST:")
bst.inorder_traversal()

print("\nSearching for 40:", bst.search(40))
print("Searching for 90:", bst.search(90))
}}}

















{{{
def iterative_deepening_dfs(graph, start, goal):
    depth = 0
    while True:
        result = depth_limited_dfs(graph, start, goal, depth, set())
        if result == "FOUND":
            return "Goal found"
        if result == "NOT_FOUND":
            print(f"Goal not found within depth {depth}")
        depth += 1

def depth_limited_dfs(graph, node, goal, depth, visited):
    if node == goal:
        return "FOUND"
    if depth == 0:
        return "NOT_FOUND"
    if depth > 0:
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                result = depth_limited_dfs(graph, neighbor, goal, depth - 1, visited)
                if result == "FOUND":
                    return "FOUND"
    return "NOT_FOUND"

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
start = 'A'
goal = 'F'

print(iterative_deepening_dfs(graph, start, goal))
}}}


