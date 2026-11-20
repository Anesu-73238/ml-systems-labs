from engine import Value

# 1. accumulation: a is used twice, so d(a*a)/da = 2a. Overwriting instead of += would give 3.
a = Value(3.0)
y = a * a
y.backward()
assert a.grad == 6.0, a.grad

# 2. diamond: z = a*b + a  ->  dz/da = b + 1, dz/db = a
a, b = Value(2.0), Value(3.0)
z = a * b + a
z.backward()
assert (a.grad, b.grad) == (4.0, 2.0), (a.grad, b.grad)

# 3. relu blocks the gradient for negative inputs
x = Value(-2.0)
r = x.relu()
r.backward()
assert r.data == 0.0 and x.grad == 0.0

# 4. the lab's expression for a single triple
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
y = ((a * b + c) ** 2).relu()
y.backward()
print("y =", y.data, " dy/da =", a.grad, " dy/db =", b.grad, " dy/dc =", c.grad)
assert (a.grad, b.grad, c.grad) == (-24.0, 16.0, 8.0)

# 5. topological order: every node comes after its parents
order = y.build_topo()
pos = {id(n): i for i, n in enumerate(order)}
assert all(pos[id(p)] < pos[id(n)] for n in order for p in n._parents)
print("nodes in graph:", len(order))

print("all backward tests passed")