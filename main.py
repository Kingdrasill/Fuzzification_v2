import matplotlib.pyplot as plt
import numpy as np
import math as mt

x = np.linspace(0, 10, 100)
y = []

for value in x:
    y.append(mt.exp(- value / 5) * mt.sin(3 * value) + 0.5 * mt.sin(value))

plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title("e^(-x/5)sin(3x) + 0.5sin(x)")
plt.grid(True)
plt.tight_layout()
plt.show()
