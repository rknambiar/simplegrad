from __future__ import annotations

class Tensor:
    """Not really a tensor"""
    counter = 1

    def __init__(self, 
                value: float) -> None:
        self.value = value
        self.parents = []
        self.name = "Tensor" + str(Tensor.counter)

        Tensor.counter += 1

    def __str__(self) -> str:
        return str(self.value)
        
    def __repr__(self) -> str:
        return f"Tensor: {self.name} value: {self.value}"

    def __add__(self, o) -> Tensor:
        output = Tensor(self.value + o.value)
        output.parents.extend([self, o])
        return output

    def __mul__(self, o) -> Tensor:
        output = Tensor(self.value * o.value)
        output.parents.extend([self, o])
        return output

    
