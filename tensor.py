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
        return f"{self.name}: {self.value:.4f}"

    def __add__(self, obj2) -> Tensor:
        obj2 = obj2 if isinstance(obj2, Tensor) else Tensor(obj2)
        output = Tensor(self.value + obj2.value, op="+")
        output.parents.extend([self, obj2])

        def backward():
            self.grad += output.grad
            obj2.grad += output.grad

        output.backward_fun = backward
        return output

    def __mul__(self, obj2) -> Tensor:
        obj2 = obj2 if isinstance(obj2, Tensor) else Tensor(obj2)
        output = Tensor(self.value * obj2.value, op="*")
        output.parents.extend([self, obj2])

        def backward():
            self.grad += obj2.value * output.grad 
            obj2.grad += self.value * output.grad

        output.backward_fun = backward
        return output

    def relu(self,) -> Tensor:
        output = Tensor(0 if self.value < 0 else self.value, op='ReLU')
        output.parents.extend([self, ])

        def backward():
            self.grad += (output.value > 0) * output.grad

        output.backward_fun = backward
        return output

    def backward(self) -> None:
        """Computes gradient from this tensor backwards"""
        def _topsort(t):
            visited = set()
            output = []
            def _run_topsort(t):
                if t not in visited:
                    visited.add(t)
                    for parent in t.parents:
                        _run_topsort(parent)
                    output.append(t)
            
            _run_topsort(t)
            return output
        
        topsort = _topsort(self)

        self.grad = 1
        for t in reversed(topsort):
            t.backward_fun()

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return o + (-self)

    def __neg__(self):
        return self * -1