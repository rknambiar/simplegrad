from __future__ import annotations

class Tensor:
    """Not really a tensor"""
    counter = 1

    def __init__(self, 
                value: float,
                op = '') -> None:
        self.value = value
        self.parents = []
        self.name = "Tensor_" + str(Tensor.counter)

        self.grad = 0.
        self.backward_fun = lambda: None
        self.op = op # For displaying which op created the tensor

        Tensor.counter += 1

    def __str__(self) -> str:
        return str(self.value)
        
    def __repr__(self) -> str:
        return f"Tensor: {self.name} value: {self.value}"

    def __add__(self, o) -> Tensor:
        output = Tensor(self.value + o.value, op="add")
        output.parents.extend([self, o])
        
        def backward():
            print(f"add backward node: {output.name}")
            self.grad += output.grad
            o.grad += output.grad

        output.backward_fun = backward

        return output

    def __mul__(self, o) -> Tensor:
        output = Tensor(self.value * o.value, op="mul")
        output.parents.extend([self, o])
        
        def backward():
            print(f"mul backward node: {output.name}")
            self.grad += output.grad * o.value
            o.grad += output.grad * self.value

        output.backward_fun = backward
        
        return output

    
