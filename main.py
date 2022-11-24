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
            
            edges.append((parent.name, curr_tensor.name, parent.grad))
            conn_str = f"From {curr_tensor.name} --> {parent.name}"
            print(conn_str)
        
        visited.add(curr_tensor)
        
    f = graphviz.Digraph('network', filename="network.gv",
                         node_attr={'style': 'filled',
                                    'color': 'lightblue2',
                                    'shape': 'box'}
    )

    for node in visited:
        label = node.name if node.op == "" else node.name + " | " + node.op
        f.node(node.name, label)

    for edge in edges:
        f.edge(edge[0], edge[1], str(edge[2]), fontcolor='red')

    f.view()

def main():
    t1 = Tensor(3.4)
    t2 = Tensor(2.6)
    t3 = Tensor(5)

    t4 = t1 + t2
    t5 = t3 * t4

    t5.backward()
    draw(t5)

if __name__ == "__main__":
    main()