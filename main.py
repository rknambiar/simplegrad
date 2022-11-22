import graphviz

from tensor import Tensor

def draw(tensor):
    visited = set()
    to_visit = [tensor]

    edges = []

    while (len(to_visit) != 0):
        curr_tensor = to_visit.pop()
        for parent in curr_tensor.parents:
            if parent not in visited:
                to_visit.append(parent)
            
            edges.append((parent.name, curr_tensor.name))
            conn_str = f"From {curr_tensor.name} --> {parent.name}"
            print(conn_str)
        
        visited.add(curr_tensor)
        
    f = graphviz.Digraph('network', filename="network.gv")
    for node in visited:
        f.node(node.name, node.name)

    for edge in edges:
        f.edge(edge[0], edge[1])

    f.view()

def main():
    t1 = Tensor(3.4)
    t2 = Tensor(2)

    t3 = t1 + t2
    t4 = t1 * t3
    t5 = t4 * t2

    draw(t5)

if __name__ == "__main__":
    main()