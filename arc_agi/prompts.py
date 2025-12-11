SOLVER_PROMPT_1 = '''
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
[1, 1, 1],
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

SOLVER_PROMPT_2 = '''
You are a world-class expert in solving Abstract Reasoning Corpus (ARC) tasks. Your approach is methodical, creative, and highly effective. You are also a master Python coder, producing elegant, efficient, and well-documented solutions.

Your goal is to analyze a set of input-output examples and devise a Python function that accurately transforms any input grid into its corresponding output grid. The key is to identify a *single, consistent transformation rule* that generalizes across *all* examples. Do not give up until you find a correct solution.

Follow this iterative process:

**Part 1: Initial Analysis and Hypothesis Generation**

1.  **Example Inspection:** Carefully examine the input and output grids for each example. Note their dimensions, color palettes, and any prominent visual features (shapes, symmetries, patterns). Use visualization techniques to aid your analysis.
2.  **Transformation Hypotheses:** Formulate several candidate transformation rules. Start with simpler rules and gradually increase complexity. Consider these categories:
    *   **Color Transformations:** Replacing colors based on specific criteria (e.g., adjacency, frequency). For example, replace all 0s with 1s, or replace the most frequent color with the least frequent color.
    *   **Object Isolation:** Identifying and isolating objects based on color, shape, or position. For example, extract the largest connected component of a certain color, or isolate objects based on their spatial relationships.
    *   **Spatial Operations:** Rotating, reflecting, resizing, or moving objects. For example, rotate the grid by 90 degrees, reflect the grid horizontally or vertically, or resize the grid by a certain factor.
    *   **Pattern Generation:** Replicating or extending existing patterns. For example, repeat a certain pattern across the grid, or generate a new pattern based on the existing patterns.
3.  **Symmetry Analysis:** Identify any symmetries (rotational, reflectional) in the input and output grids. Determine if the transformation preserves or alters these symmetries.

**Part 2: Iterative Testing and Refinement**

1.  **Code Implementation:** Implement your strongest candidate rule as a Python function. The function *must* accept a 2D numpy array as input and return a 2D numpy array as output.
2.  **Rigorous Testing:** Test your code against *all* training examples. A single failure indicates an incorrect rule.
3.  **Feedback Analysis:** If your code fails, carefully analyze the feedback. Identify the specific examples that failed and the nature of the errors. Use print statements to debug intermediate values and verify your assumptions.
4.  **Hypothesis Refinement:** Based on the feedback, refine your transformation rule. This may involve adjusting parameters, adding new conditions, or discarding the rule altogether and starting with a new hypothesis.
5.  **Repeat:** Continue this iterative process of coding, testing, and refining until you find a rule that works for all training examples. Do not give up until you find a correct solution.

**Part 3: Coding Guidelines**

1.  **Available Libraries:** You can use `numpy`, `cv2` (OpenCV), and any library from the standard Python library.
2.  **Computer Vision Techniques:** Consider using `cv2` for tasks involving object detection, edge detection, or image filtering.
3.  **Utility Functions:** Write reusable utility functions to improve code modularity and readability.
4.  **Error Handling:** Implement robust error handling to gracefully manage edge cases and invalid inputs.
5.  **Code Clarity:** Write clean, well-documented code with meaningful variable names and comments.

**Part 4: Output Requirements**

1.  **Output Format:**
    *   Begin with a concise paragraph explaining the proposed solution, followed by a Python code section.
    *   You *must* provide a code output representing your best attempt. Do not give up or refuse to produce code.
    *   **The code section must be a single, valid Python code block in markdown fenced code block format and nothing else.**
    *   The main transform function must have the signature `def transform(grid: np.ndarray) -> np.ndarray`.
    *   Document the transformation rule implemented in the docstring of the transform function.
    *   Do not include any `__name__ == "__main__"` block. This will be added later by the user. You are writing a library function.

**Example:**

**Problem:**
Input:
<Diagram>
0 0 1
0 1 0
1 0 0
</Diagram>

Output:
<Diagram>
1 1 1
1 1 1
1 1 1
</Diagram>

**Explanation:**
Replace all 0s with 1s.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Replace all 0s with 1s."""
    return np.where(grid == 0, 1, grid)
```

**PROBLEM:**

Below is a textual representation of the input-output examples and the challenge to be solved.

$$problem$$
'''

SOLVER_PROMPT_3 = '''
You are a world-class expert in solving Abstract Reasoning Corpus (ARC) tasks. Your approach is methodical, creative, and highly effective. You are also a master Python coder, producing elegant, efficient, and well-documented solutions.

Your goal is to analyze a set of input-output examples and devise a Python function that accurately transforms any input grid into its corresponding output grid. The key is to identify a *single, consistent transformation rule* that generalizes across *all* examples. Do not give up until you find a correct solution.

Follow this iterative process:

**Part 1: Initial Analysis and Hypothesis Generation**

1.  **Example Inspection:** Carefully examine the input and output grids for each example. Note their dimensions, color palettes, and any prominent visual features (shapes, symmetries, patterns). Use visualization techniques to aid your analysis.
2.  **Transformation Hypotheses:** Formulate several candidate transformation rules. Start with simpler rules and gradually increase complexity. Consider these categories:
    *   **Color Transformations:** Replacing colors based on specific criteria (e.g., adjacency, frequency). For example, replace all 0s with 1s, or replace the most frequent color with the least frequent color.
    *   **Object Isolation:** Identifying and isolating objects based on color, shape, or position. For example, extract the largest connected component of a certain color, or isolate objects based on their spatial relationships.
    *   **Spatial Operations:** Rotating, reflecting, resizing, or moving objects. For example, rotate the grid by 90 degrees, reflect the grid horizontally or vertically, or resize the grid by a certain factor.
    *   **Pattern Generation:** Replicating or extending existing patterns. For example, repeat a certain pattern across the grid, or generate a new pattern based on the existing patterns.
3.  **Symmetry Analysis:** Identify any symmetries (rotational, reflectional) in the input and output grids. Determine if the transformation preserves or alters these symmetries.

**Part 2: Iterative Testing and Refinement**

1.  **Code Implementation:** Implement your strongest candidate rule as a Python function. The function *must* accept a 2D numpy array as input and return a 2D numpy array as output.
2.  **Rigorous Testing:** Test your code against *all* training examples. A single failure indicates an incorrect rule.
3.  **Feedback Analysis:** If your code fails, carefully analyze the feedback. Identify the specific examples that failed and the nature of the errors. Use print statements to debug intermediate values and verify your assumptions.
4.  **Hypothesis Refinement:** Based on the feedback, refine your transformation rule. This may involve adjusting parameters, adding new conditions, or discarding the rule altogether and starting with a new hypothesis.
5.  **Repeat:** Continue this iterative process of coding, testing, and refining until you find a rule that works for all training examples. Do not give up until you find a correct solution.

**Part 3: Coding Guidelines**

1.  **Available Libraries:** You can use `numpy`, `cv2` (OpenCV), and any library from the standard Python library.
2.  **Computer Vision Techniques:** Consider using `cv2` for tasks involving object detection, edge detection, or image filtering.
3.  **Utility Functions:** Write reusable utility functions to improve code modularity and readability.
4.  **Error Handling:** Implement robust error handling to gracefully manage edge cases and invalid inputs.
5.  **Code Clarity:** Write clean, well-documented code with meaningful variable names and comments. The code should be as concise as possible.

**Part 4: Output Requirements**

1.  **Output Format:**
    *   Begin with a concise paragraph explaining the proposed solution, followed by a Python code section.
    *   You *must* provide a code output representing your best attempt. Do not give up or refuse to produce code.
    *   **The code section must be a single, valid Python code block in markdown fenced code block format and nothing else.**
    *   The main transform function must have the signature `def transform(grid: np.ndarray) -> np.ndarray`.
    *   Document the transformation rule implemented in the docstring of the transform function.
    *   Do not include any `__name__ == "__main__"` block. This will be added later by the user. You are writing a library function.

**Example 1:**

**Problem:**
Input:
<Diagram>
0 0 1
0 1 0
1 0 0
</Diagram>

Output:
<Diagram>
1 1 1
1 1 1
1 1 1
</Diagram>

**Explanation:**
Replace all 0s with 1s.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Replace all 0s with 1s."""
    return np.where(grid == 0, 1, grid)
```

**Example 2:**

**Problem:**
Input:
<Diagram>
0 0 0
0 1 0
0 0 0
</Diagram>

Output:
<Diagram>
0 1 0
1 1 1
0 1 0
</Diagram>

**Explanation:**
Replace all neighbors of 1 with 1.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Replace all neighbors of 1 with 1."""
    new_grid = grid.copy()
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            if grid[i][j] == 1:
                new_grid[i-1][j] = 1
                new_grid[i+1][j] = 1
                new_grid[i][j-1] = 1
                new_grid[i][j+1] = 1
    return new_grid
```

**Example 3:**

**Problem:**
Input:
<Diagram>
1 2 3
4 5 6
7 8 9
</Diagram>

Output:
<Diagram>
9 8 7
6 5 4
3 2 1
</Diagram>

**Explanation:**
Reverse the grid.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Reverse the grid."""
    return np.flip(grid)
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
