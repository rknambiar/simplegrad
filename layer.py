import random

from tensor import  Tensor

class Neuron:
    def __init__(self, in_channels, activation="relu") -> None:
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(in_channels)]
        self.b = Tensor(0.)
        self.activation = activation

    def __call__(self, x) -> Tensor:
        output = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return output.relu() if self.activation == "relu" else output

    def parameters(self):
        return self.w + [self.b]

class Linear:
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                activation: str = "relu") -> None:
        self.name = "Linear"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.neurons = [Neuron(in_channels, activation) for _ in range(out_channels)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output
    
    def __str__(self) -> str:
        return f"Linear layer [{self.in_channels, self.out_channels, self.activation}]"

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
class MLP:
    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, inp: int, outp: int, activation="relu") -> None:
        self.layers.append(Linear(inp, outp, activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __str__(self) -> str:
        return f"MLP: {', '.join(str(l) for l in self.layers)}"
        
    def parameters(self) -> list[Tensor]:
        return [param for layer in self.layers for param in layer.parameters()]

    def zero_grad(self) -> None:
        for weight in self.parameters():
            weight.grad = 0.