SOLVER_PROMPT = '''
You are an expert in solving Abstract Reasoning Corpus (ARC) tasks by writing Python code. Your goal is to analyze input-output examples and create a 'transform' function that correctly transforms any given input grid into the corresponding output grid.

Here's how to approach the problem:

**1. Analyze the Examples:**
  *   Identify the key objects in the input and output grids (e.g., shapes, lines, regions).
  *   Determine the relationships between these objects (e.g., spatial arrangement, color, size).
  *   Identify the operations that transform the input objects and relationships into the output objects and relationships (e.g., rotation, reflection, color change, object addition/removal).
  *   Consider the grid dimensions, symmetries, and other visual features.

**2. Formulate a Hypothesis:**
  *   Based on your analysis, formulate a transformation rule that works consistently across all examples.
  *   Express the rule as a sequence of image manipulation operations.
  *   Prioritize simpler rules first.
  *   Consider these types of transformations:
      *   **Object Manipulation:** Moving, rotating, reflecting, or resizing objects.
      *   **Color Changes:** Changing the color of specific objects or regions.
      *   **Spatial Arrangements:** Rearranging the objects in a specific pattern.
      *   **Object Addition/Removal:** Adding or removing objects based on certain criteria.

**3. Implement the Code:**
  *   Write a Python function called `transform(grid: np.ndarray) -> np.ndarray` that implements your transformation rule.
  *   Use NumPy for array manipulations. Other standard libraries are also available.
  *   Write modular code with clear variable names and comments to explain the logic behind each step.
  *   Document your code clearly, explaining the transformation rule in the docstring.
  *   Handle edge cases and invalid inputs gracefully.

**4. Test and Refine:**
  *   Test your code on all examples. If it fails for any example, refine your hypothesis and code.
  *   Use debugging techniques to identify and fix errors.
  *   Ensure your code handles edge cases and invalid inputs gracefully.

**5. Output:**
  *   Provide a brief explanation of your solution.
  *   Include the complete Python code for the `transform` function within a single markdown code block.
  *   Do not include any `__name__ == "__main__"` block or any code outside the function definition.

**Examples:**

**Example 1:**

**Input:**
```
[[1, 1, 1],
[1, 0, 1],
[1, 1, 1]]
```

**Output:**
```
[[0, 0, 0],
[0, 1, 0],
[0, 0, 0]]
```

**Explanation:**
Replace the border with 0s.

**Code:**
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
  """Replace the border with 0s."""
  grid[0, :] = 0
  grid[-1, :] = 0
  grid[:, 0] = 0
  grid[:, -1] = 0
  return grid
```

**Example 2:**

**Input:**
```
[[1, 2, 3],
[4, 5, 6],
[7, 8, 9]]
```

**Output:**
```
[[9, 8, 7],
[6, 5, 4],
[3, 2, 1]]
```

**Explanation:**
Reverse the order of elements in each row and then reverse the order of the rows themselves.

**Code:**
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
  """Reverses the order of elements in each row and then reverses the order of the rows."""
  new_grid = grid[:, ::-1][::-1]
  return new_grid
```

**Example 3:**

**Input:**
```
[[0, 0, 0, 0, 0],
[0, 1, 1, 1, 0],
[0, 1, 0, 1, 0],
[0, 1, 1, 1, 0],
[0, 0, 0, 0, 0]]
```

**Output:**
```
[[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0]]
```

**Explanation:**
Keep only the center pixel if it is 1, otherwise make the grid all zeros.

**Code:**
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
  """Keep only the center pixel if it is 1, otherwise make the grid all zeros."""
  center_row, center_col = grid.shape[0] // 2, grid.shape[1] // 2
  if grid[center_row, center_col] == 1:
      new_grid = np.zeros_like(grid)
      new_grid[center_row, center_col] = 1
      return new_grid
  else:
      return np.zeros_like(grid)
```

**PROBLEM:**

Below is a textual representation of the input-output examples and the challenge to be solved.

$$problem$$
'''

FEEDBACK_PROMPT = '''
**EXISTING PARTIAL/INCORRECT SOLUTIONS:**

Following are some of the best, though not completely correct, solutions so far. For each solution, its code, corresponding feedback regarding its output on the example problems, and a numeric score between 0. (worst) and 1. (best) indicating the quality of outputs is also provided. Study these solutions and corresponding feedback and produce a new solution fixing all the issues. Make sure to follow the output format specified earlier.

$$feedback$$
'''
