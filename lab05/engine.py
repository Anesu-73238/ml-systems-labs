import math


class Value:
    """A scalar that remembers how it was computed.

    data      : the number
    grad      : d(output)/d(this node), filled in by backward()
    _parents  : the nodes this one was computed from (graph edges)
    _backward : the local derivative rule, which pushes this node's grad into its parents
    _op       : the operation name, for inspecting the graph
    """

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(parents)
        self._backward = lambda: None
        self._op = op

    # ---- the four required operations: + * ** relu ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad      # d(a+b)/da = 1
            other.grad += out.grad     # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad   # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "exponent must be a plain number"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad   # d(a^k)/da = k a^(k-1)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad              # 1 if positive, else 0
        out._backward = _backward
        return out

    # ---- extra primitives, needed for sigmoid and BCE in Step 5 ----
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad                    # d(e^a)/da = e^a
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad           # d(ln a)/da = 1/a
        out._backward = _backward
        return out

    # ---- backward pass ----
    def build_topo(self):
        """Topological order of the graph that ends at this node: every node appears
        AFTER all of its parents. Iterative DFS, so long graphs can't hit the recursion limit."""
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)          # all parents are already in topo
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))     # come back to this node after its parents
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        return topo

    def backward(self):
        topo = self.build_topo()
        self.grad = 1.0                    # d(output)/d(output) = 1
        for node in reversed(topo):        # outputs first, inputs last
            node._backward()               # each rule does `+=` into its parents

    # ---- conveniences built from the primitives (they also build the graph) ----
    def __neg__(self):            return self * -1
    def __sub__(self, other):     return self + (-other)
    def __radd__(self, other):    return self + other
    def __rsub__(self, other):    return (-self) + other
    def __rmul__(self, other):    return self * other
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"


if __name__ == "__main__":
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    y = ((a * b + c) ** 2).relu()
    print("value :", y.data)                        # (2*-3 + 10)^2 = 16
    print("node  :", y)
    print("parent:", y._parents[0])                 # the **2 node
    print("graph walk (output -> inputs):")
    node = y
    while node._parents:
        print("  ", node._op, "<-", [p._op or "leaf" for p in node._parents])
        node = node._parents[0]