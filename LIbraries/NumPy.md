# NumPy Master Notes

> [!tip] Quick Import
> 
> ```python
> import numpy as np
> ```

---

## Array Creation & Basics

### Creating Arrays

```python
# Basic arrays
a = np.array([1, 2, 3])                    # 1D array
b = np.array([[1, 2], [3, 4]])             # 2D array
c = np.array([1, 2, 3], dtype=float)       # Specify data type
```

### Smart Creation Methods

|Method|Purpose|Example|
|---|---|---|
|`np.arange(start, stop, step)`|Range of numbers|`np.arange(0, 10, 2)` → `[0,2,4,6,8]`|
|`np.linspace(start, stop, num)`|Evenly spaced|`np.linspace(0, 1, 5)` → `[0, 0.25, 0.5, 0.75, 1]`|
|`np.ones((rows, cols))`|Array of ones|`np.ones((2, 3))`|
|`np.zeros((rows, cols))`|Array of zeros|`np.zeros((2, 3))`|
|`np.eye(n)`|Identity matrix|`np.eye(3)`|

> [!example] Power Move
> 
> ```python
> # Create and reshape in one line
> a = np.arange(0, 15).reshape(3, 5)
> # Creates: [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14]]
> ```

### Array Properties

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape)    # (2, 3) - dimensions
print(a.ndim)     # 2 - number of dimensions  
print(a.dtype)    # int64 - data type
print(a.size)     # 6 - total elements
```

---

## Array Indexing & Slicing

### Basic Indexing

```python
a = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])

# Basic slicing
print(a[1, 2])        # 6 - element at row 1, col 2
print(a[:, 1:3])      # All rows, columns 1-2
print(a[1:, :2])      # Rows 1+, first 2 columns
```

### Fancy Indexing

```python
# Boolean indexing
mask = a > 5
print(a[mask])        # [6, 7, 8, 9, 10, 11]

# Advanced indexing
rows = [0, 2]
cols = [1, 3]
print(a[rows, cols])  # Elements at (0,1) and (2,3)
```

> [!warning] View vs Copy
> 
> ```python
> s = a[:, 1:3]    # This is a VIEW
> s[:] = 10        # This modifies original array 'a'!
> 
> c = a.copy()     # This is a COPY
> c[:] = 99        # This doesn't affect 'a'
> ```

### Ellipsis Magic

```python
c = np.array([[[1, 2], [3, 4]], 
              [[5, 6], [7, 8]]])

print(c[1, ...])     # Same as c[1, :, :]
print(c[..., 1])     # Same as c[:, :, 1]
```

---

## Mathematical Operations

### Element-wise Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)      # [5, 7, 9]
print(a * b)      # [4, 10, 18] - element-wise
print(a ** 2)     # [1, 4, 9]
print(np.sqrt(a)) # [1, 1.414, 1.732]
```

### Matrix Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)           # Matrix multiplication
print(np.dot(A, B))    # Same as above
print(A.T)             # Transpose
```

---

## Statistical Functions

### Basic Stats

|Function|Purpose|Axis Parameter|
|---|---|---|
|`np.sum(a)`|Total sum|`axis=0` (columns), `axis=1` (rows)|
|`np.mean(a)`|Average|Same as sum|
|`np.std(a)`|Standard deviation|Same as sum|
|`np.min(a)`, `np.max(a)`|Min/Max values|Same as sum|

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(np.sum(a))           # 21 - total sum
print(np.sum(a, axis=0))   # [5, 7, 9] - column sums
print(np.sum(a, axis=1))   # [6, 15] - row sums
```

### Advanced Stats

```python
a = np.array([1, 3, 2, 5, 4])

print(np.median(a))        # 3.0
print(np.percentile(a, 75)) # 4.0
print(np.argmax(a))        # 3 - index of max element
print(np.argmin(a))        # 0 - index of min element
```

---

## Array Manipulation

### Shape Changes

```python
a = np.arange(12)

print(a.reshape(3, 4))     # 3x4 matrix
print(a.reshape(-1, 2))    # Auto-calculate rows, 2 columns
print(a.ravel())           # Flatten to 1D
```

### Combining Arrays

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stack
print(np.vstack([a, b]))   # [[1,2], [3,4], [5,6], [7,8]]

# Horizontal stack  
print(np.hstack([a, b]))   # [[1,2,5,6], [3,4,7,8]]

# Concatenate with axis
print(np.concatenate([a, b], axis=0))  # Same as vstack
```

### Splitting Arrays

```python
a = np.arange(12).reshape(3, 4)

# Split horizontally
left, right = np.hsplit(a, 2)

# Split at specific indices  
parts = np.hsplit(a, [1, 3])  # Split after columns 1 and 3
```

---

## Random Numbers

```python
# Set up random generator
rng = np.random.default_rng(42)  # Seed for reproducibility

# Generate random arrays
print(rng.random((2, 3)))           # Uniform [0, 1)
print(rng.integers(0, 10, (2, 3)))  # Random integers
print(rng.normal(0, 1, (2, 3)))     # Normal distribution
```

---

## Boolean Operations & Filtering

### Logical Operations

```python
a = np.array([1, 2, 3, 4, 5])

# Boolean conditions
print(a > 3)                # [False, False, False, True, True]
print(np.any(a > 3))        # True - any element > 3?
print(np.all(a > 0))        # True - all elements > 0?
```

### Where Magic

```python
a = np.array([-1, 2, -3, 4, -5])

# Replace negatives with 0, keep positives
result = np.where(a > 0, a, 0)  # [0, 2, 0, 4, 0]

# Multiple conditions
result = np.where((a > 0) & (a < 4), a, -999)
```

---

## Advanced Tricks

### Broadcasting

```python
# Array shapes: (3,1) and (4,) broadcast to (3,4)
a = np.array([[1], [2], [3]])    # (3, 1)
b = np.array([10, 20, 30, 40])   # (4,)

result = a + b  # Broadcasts to (3, 4) array
```

### Fancy Functions

```python
# fromfunction - create arrays using functions
def f(i, j):
    return 10 * i + j

grid = np.fromfunction(f, (3, 4), dtype=int)
# [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
```

### Color Palette Indexing

```python
# Define color palette
palette = np.array([[0, 0, 0],      # black
                    [255, 0, 0],    # red  
                    [0, 255, 0],    # green
                    [0, 0, 255]])   # blue

# Image with color indices
image = np.array([[0, 1], [2, 3]])

# Convert to RGB
rgb_image = palette[image]  # Shape: (2, 2, 3)
```

---

## Performance Tips

> [!tip] Speed Hacks
> 
> - Use vectorized operations instead of loops
> - Prefer views over copies when possible
> - Use appropriate dtypes (int8 vs int64)
> - Pre-allocate arrays with `np.zeros()` or `np.empty()`

### Vectorization Example

```python
# ❌ Slow - Python loop
result = []
for x in range(1000000):
    result.append(x ** 2)

# ✅ Fast - NumPy vectorized
x = np.arange(1000000)
result = x ** 2
```

---

## Common Patterns

### Iteration Patterns

```python
a = np.array([[1, 2], [3, 4]])

# Iterate over rows
for row in a:
    print(row)

# Iterate over all elements
for element in a.flat:
    print(element)

# Enumerate with indices
for i, row in enumerate(a):
    print(f"Row {i}: {row}")
```

### Axis Operations Cheat Sheet

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Remember: axis=0 goes DOWN (rows), axis=1 goes ACROSS (columns)
print(a.sum(axis=0))    # [5, 7, 9] - sum each column
print(a.sum(axis=1))    # [6, 15] - sum each row
```

---

## Pro Tips

> [!success] Master Moves
> 
> 1. **Use `axis` parameter** for operations on specific dimensions
> 2. **Master broadcasting** for efficient operations
> 3. **Understand views vs copies** to avoid bugs
> 4. **Use boolean indexing** for filtering
> 5. **Vectorize everything** for performance

> [!example] One-Liner Magic
> 
> ```python
> # Normalize array to [0, 1] range
> normalized = (a - a.min()) / (a.max() - a.min())
> 
> # Find indices of top 3 values
> top3_indices = np.argpartition(a, -3)[-3:]
> 
> # Replace outliers with median
> median = np.median(a)
> a[np.abs(a - median) > 2 * np.std(a)] = median
> ```