# nanograd.py
# Corrected minimal autograd engine (NanoGrad)

import random
import math
import numpy as np

class Value:
    """
    A scalar value with automatic differentiation support.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)  # parent nodes
        self._op = _op               # operation that produced this node (for debug)
        self._backward = lambda: None  # function to propagate gradients to parents
        # optional name for debugging
        self.name = ''

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ---------- basic ops ----------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**(-1)

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports int/float powers for simplicity"
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---------- backward (autodiff) ----------
    def backward(self):
        """
        Run backpropagation to compute gradients for all nodes that affect this node.
        """
        # topological order via DFS
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        # reset gradients
        for node in topo:
            node.grad = 0.0

        # seed gradient at output
        self.grad = 1.0

        # traverse in reverse topological order
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0

    # convenience to convert python numbers to Value
    @staticmethod
    def from_list(lst):
        return [Value(x) for x in lst]


# ----------------- small neural network using Value -----------------
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        # weighted sum
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, nin, nouts):
        # nouts is list of neurons per layer, e.g. [16, 16, 1]
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            layer = [Neuron(sz[i]) for _ in range(nouts[i])]
            self.layers.append(layer)

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = [n(out) for n in layer]  # compute all neurons in this layer
            if len(out) == 1:
                out = out[0]  # unwrap single output neuron
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            for n in layer:
                params += n.parameters()
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ----------------- basic tests & training routine -----------------
def test_basic_ops():
    print("\n--- Test 1: Basic Operations ---")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x**2
    print(f"x = {x.data}, y = {y.data}")
    print(f"z = x*y + x^2 = {z.data}")
    z.backward()
    print(f"dz/dx = {x.grad} (expected: {y.data + 2*x.data})")
    print(f"dz/dy = {y.grad} (expected: {x.data})")

def test_mlp_training():
    print("\n--- Test 2: Small Neural Network ---")
    # toy dataset: y = 2*x + 3 (scalar)
    xs = [Value(x) for x in [0.0, 1.0, 2.0, 3.0]]
    ys = [Value(2.0 * x + 3.0) for x in [0.0, 1.0, 2.0, 3.0]]

    net = MLP(1, [8, 1])
    params = net.parameters()

    steps = 20
    lr = 0.05
    for i in range(steps):
        losses = []
        for x_val, y_val in zip(xs, ys):
            out = net([x_val])  # single output neuron
            diff = out - y_val
            losses.append(diff * diff)
        loss = sum(losses, Value(0.0))

        loss.backward()
        for p in params:
            p.data -= lr * p.grad

        if i % (steps//4) == 0 or i == steps-1:
            print(f"Step {i}: loss = {loss.data:.4f}")

if __name__ == "__main__":
    print("="*60)
    print("NanoGrad - Autograd Engine Test")
    print("="*60)
    test_basic_ops()
    test_mlp_training()
    print("\nNOTE: This code is intended to be a minimal autograd engine.")
    print("Use validator.py to run the official test suite.")
