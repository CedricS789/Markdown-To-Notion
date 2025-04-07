# Meeting Notes - Project Alpha

## Introduction

Discussed the progress on Project Alpha. Key points below.

## Key Formulas

The fundamental equation is Einstein's famous $E=mc^2$. This relates energy (E), mass (m), and the speed of light (c).

We also reviewed the quadratic formula:

$$
ax^2 + bx + c = 0 \implies x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

Make sure to check the discriminant $\Delta = b^2 - 4ac$.

## Action Items

* Review the derivation for $e^{i\pi} + 1 = 0$.
* Implement the algorithm using `code_inline`.

## Code Example

```python
import numpy as np

def solve_quadratic(a, b, c):
  """Solves ax^2 + bx + c = 0"""
  delta = b**2 - 4*a*c
  if delta < 0:
    return None # No real solutions
  x1 = (-b + np.sqrt(delta)) / (2*a)
  x2 = (-b - np.sqrt(delta)) / (2*a)
  return x1, x2

print(solve_quadratic(1, -3, 2))
```
