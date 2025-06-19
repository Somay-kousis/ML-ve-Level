# Pandas Complete Reference Notes

## Table of Contents

- [[#Basic Data Structures]]
- [[#DataFrame Creation]]
- [[#Data Inspection]]
- [[#Data Selection & Indexing]]
- [[#Filtering]]
- [[#Missing Data]]
- [[#Data Manipulation]]
- [[#String Operations]]
- [[#Combining DataFrames]]
- [[#GroupBy Operations]]
- [[#Reshaping Data]]
- [[#Time Series]]
- [[#Categorical Data]]
- [[#File I/O]]

---

## Basic Data Structures

### Series

```python
import pandas as pd
import numpy as np

# Creating a Series
series = pd.Series([1, 2, 3, np.nan, 6, 8])
```

**Key Points:**

- One-dimensional labeled array
- Can hold any data type
- Index is automatically created (0, 1, 2, ...)

### DataFrame

```python
# From arrays with date index
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=("A","B","C","D"))

# From dictionary
df2 = pd.DataFrame({
    "A": 1.0,
    "B": pd.Timestamp("20130102"),
    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    "D": np.array([3] * 4, dtype="int32"),
    "E": pd.Categorical(["test", "train", "test", "train"]),
    "F": "foo",
})
```

**Key Methods:**

- `df.dtypes` - Check data types
- `df.to_numpy()` - Convert to NumPy array

---

## Data Inspection

### Basic Info

```python
df.head()        # First 5 rows (default)
df.head(3)       # First 3 rows
df.tail(3)       # Last 3 rows
df.describe()    # Statistical summary
df.info()        # Data types and memory usage
df.shape         # Dimensions (rows, columns)
```

### Structure Operations

```python
df.T                                    # Transpose
df.sort_index(axis=1, ascending=False)  # Sort by column names
df.sort_values(by="B")                  # Sort by column values
```

**Axis Reference:**

- `axis=0` → Rows (index)
- `axis=1` → Columns

---

## Data Selection & Indexing

### Label-based vs Position-based

|Method|Type|Single|Multiple|Description|
|---|---|---|---|---|
|`at()`|Label|✓|✗|Single cell by label|
|`iat()`|Position|✓|✗|Single cell by position|
|`loc()`|Label|✓|✓|Multiple cells by label|
|`iloc()`|Position|✓|✓|Multiple cells by position|

```python
df = pd.DataFrame({"A": [10, 20], "B": [30, 40]}, index=["x", "y"])

# Single cell access
print(df.at["x", "A"])      # 10
print(df.iat[0, 0])         # 10

# Column selection
df["A"]                     # Single column
df[["A", "B"]]             # Multiple columns

# Row slicing with dates
df["20130102":"20130104"]   # Includes both endpoints

# Label-based selection
df.loc["20130102":"20130104", ["A", "B"]]

# Position-based selection  
df.iloc[3:5, 0:2]          # Rows 3-4, Columns 0-1
```

---

## Filtering

### Boolean Indexing

```python
# Row-wise filter
df[df["A"] > 0]

# Element-wise filter  
df[df > 0]

# Using isin()
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
df2[df2["E"].isin(["two", "four"])]
```

**Common Filter Patterns:**

- `df[df['col'] > value]`
- `df[df['col'].between(a, b)]`
- `df[df['col'].str.contains('pattern')]`
- `df[df['col'].isnull()]`

---

## Missing Data

### Detection

```python
pd.isna(df)         # True for NaN values
df.isnull()         # Same as isna()
df.notnull()        # Opposite of isnull()
```

### Handling Missing Data

```python
# Drop rows/columns with NaN
df.dropna(how="any")        # Drop if ANY NaN in row
df.dropna(how="all")        # Drop if ALL NaN in row
df.dropna(axis=1)           # Drop columns with NaN

# Fill missing values
df.fillna(value=5)          # Fill with constant
df.fillna(method='ffill')   # Forward fill
df.fillna(method='bfill')   # Backward fill
df.fillna(df.mean())        # Fill with mean
```

---

## Data Manipulation

### Aggregation vs Transformation

```python
# Aggregation - changes shape, returns summary
df.agg(lambda x: np.mean(x) * 5.6)      # Column-wise operation

# Transformation - preserves shape
df.transform(lambda x: x * 101.2)       # Element-wise operation
```

### Statistical Operations

```python
df.mean(axis=0)     # Column means (default)
df.mean(axis=1)     # Row means
df.sum()            # Column sums
df.std()            # Standard deviation
df.var()            # Variance
df.corr()           # Correlation matrix
```

### Value Counts

```python
s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()    # Frequency of each value
```

### Shifting Data

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates)
s.shift(2)          # Shift values down by 2 positions
s.shift(-1)         # Shift values up by 1 position
```

---

## String Operations

```python
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])

# String methods (use .str accessor)
s.str.lower()           # Convert to lowercase
s.str.upper()           # Convert to uppercase
s.str.len()             # String length
s.str.contains('a')     # Contains pattern
s.str.startswith('C')   # Starts with pattern
s.str.replace('a', 'X') # Replace pattern
```

**Note:** Always use `.str` accessor for string operations on Series

---

## Combining DataFrames

### Concatenation

```python
# Vertical concatenation (stacking)
df1 = df[:3]
df2 = df[3:7] 
df3 = df[7:]
pieces = [df1, df2, df3]
pd.concat(pieces)                    # Default: axis=0 (vertical)
pd.concat(pieces, axis=1)            # Horizontal concatenation
pd.concat(pieces, ignore_index=True) # Reset index
```

### Merging (SQL-like joins)

```python
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})

pd.merge(left, right, on="key")              # Inner join (default)
pd.merge(left, right, on="key", how="left")  # Left join
pd.merge(left, right, on="key", how="outer") # Outer join
```

**Join Types:**

- `inner` - Only matching keys
- `left` - All from left, matching from right
- `right` - All from right, matching from left
- `outer` - All keys from both

---

## GroupBy Operations

### The Split-Apply-Combine Process

1. **Split** - Divide data into groups
2. **Apply** - Apply function to each group
3. **Combine** - Combine results

```python
df = pd.DataFrame({
    "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
    "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
    "C": np.random.randn(8),
    "D": np.random.randn(8),
})

# Single column grouping
df.groupby("A")[["C", "D"]].sum()

# Multiple column grouping  
df.groupby(["A", "B"]).sum()

# Common aggregations
df.groupby("A").agg({
    'C': 'mean',
    'D': ['sum', 'count', 'std']
})
```

**Common GroupBy Methods:**

- `.sum()`, `.mean()`, `.count()`
- `.min()`, `.max()`, `.std()`
- `.agg()` - Custom aggregations
- `.apply()` - Custom functions

---

## Reshaping Data

### MultiIndex and Stacking

```python
# Create MultiIndex
arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]
index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])

# Stack/Unstack operations
stacked = df.stack(future_stack=True)    # Columns → Index levels
stacked.unstack(0)                       # Level 0 → Columns  
stacked.unstack(1)                       # Level 1 → Columns
stacked.unstack()                        # Last level → Columns
```

### Pivot Tables

```python
df = pd.DataFrame({
    "A": ["one", "one", "two", "three"] * 3,
    "B": ["A", "B", "C"] * 4, 
    "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
    "D": np.random.randn(12),
    "E": np.random.randn(12),
})

# Create pivot table
pd.pivot_table(df, 
               values="D",           # Values to aggregate
               index=["A", "B"],     # Row groupers
               columns=["C"],        # Column groupers
               aggfunc='mean')       # Aggregation function
```

**Pivot vs GroupBy:**

- **Pivot** - Reshape data, specific format
- **GroupBy** - More flexible aggregation

---

## Time Series

### Date Ranges and Indexing

```python
# Create date range
rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)

# Frequency aliases
# 'D' - Daily, 'H' - Hourly, 'M' - Monthly
# 'S' - Seconds, 'T' or 'min' - Minutes
```

### Resampling

```python
# Create high-frequency data
rng = pd.date_range("1/1/2012", periods=100, freq="s")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

# Downsample to 5-minute intervals
ts.resample("5Min").sum()    # Sum within each 5-min window
ts.resample("5Min").mean()   # Average within each 5-min window
```

### Time Zones

```python
ts_utc = ts.tz_localize("UTC")           # Set timezone
ts_eastern = ts_utc.tz_convert("US/Eastern")  # Convert timezone
```

**Common Resampling Rules:**

- Downsampling: `'5Min'`, `'H'`, `'D'`, `'M'`
- Upsampling: Use `.asfreq()` or `.resample().interpolate()`

---

## Categorical Data

### Creating Categories

```python
df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6], 
    "raw_grade": ["a", "b", "b", "a", "a", "e"]
})

# Convert to category
df["grade"] = df["raw_grade"].astype("category")
```

### Category Operations

```python
# Rename categories
new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)

# Set all possible categories (including unused ones)
df["grade"] = df["grade"].cat.set_categories([
    "very bad", "bad", "medium", "good", "very good"
])

# Sorting respects category order
df.sort_values(by="grade")

# Count including zero counts
df.groupby("grade", observed=False).size()
```

**Benefits of Categories:**

- Memory efficient for repeated strings
- Enables custom sorting order
- Statistical operations respect categories

---

## File I/O

### CSV Operations

```python
# Write CSV
df.to_csv("foo.csv")
df.to_csv("foo.csv", index=False)        # Without row index
df.to_csv("foo.csv", sep=';')            # Custom separator

# Read CSV  
pd.read_csv("foo.csv")
pd.read_csv("foo.csv", index_col=0)      # Use first column as index
pd.read_csv("foo.csv", parse_dates=True) # Parse date columns
```

### Parquet (Recommended for large datasets)

```python
df.to_parquet("foo.parquet")
pd.read_parquet("foo.parquet")
```

### Excel Operations

```python
# Write Excel
df.to_excel("foo.xlsx", sheet_name="Sheet1")
df.to_excel("foo.xlsx", sheet_name="Sheet1", index=False)

# Read Excel
pd.read_excel("foo.xlsx", "Sheet1", 
              index_col=None,           # Don't use any column as index
              na_values=["NA"])         # Treat "NA" strings as NaN
```

### Other Formats

```python
# JSON
df.to_json("data.json")
pd.read_json("data.json")

# SQL (requires SQLAlchemy)
df.to_sql("table_name", connection)
pd.read_sql("SELECT * FROM table", connection)

# HDF5 (requires PyTables)
df.to_hdf("data.h5", key="df")
pd.read_hdf("data.h5", "df")
```

---

## Advanced Tips & Best Practices

### Memory Optimization

```python
# Check memory usage
df.info(memory_usage='deep')

# Optimize data types
df['category_col'] = df['category_col'].astype('category')
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
```

### Performance Tips

- Use `.loc` and `.iloc` for explicit indexing
- Vectorized operations are faster than loops
- Use `.query()` for complex filtering: `df.query('A > 0 and B < 5')`
- Chain operations with method chaining: `df.dropna().groupby('col').sum()`

### Common Gotchas

- **SettingWithCopyWarning**: Use `.copy()` when creating DataFrames from slices
- **Index alignment**: Operations align on index automatically
- **NaN handling**: Most operations propagate NaN values
- **String operations**: Always use `.str` accessor

---

## Quick Reference Cheat Sheet

|Operation|Code|Description|
|---|---|---|
|**Selection**|`df['col']`|Select column|
||`df.loc[row, col]`|Label-based selection|
||`df.iloc[1:3, 0:2]`|Position-based selection|
|**Filtering**|`df[df['col'] > 5]`|Boolean filtering|
||`df.query('col > 5')`|Query syntax|
|**Aggregation**|`df.groupby('col').sum()`|GroupBy operations|
||`df.agg({'col': 'mean'})`|Custom aggregation|
|**Missing Data**|`df.dropna()`|Remove NaN|
||`df.fillna(0)`|Fill NaN with value|
|**Reshaping**|`df.pivot_table()`|Create pivot table|
||`df.melt()`|Unpivot data|
|**I/O**|`pd.read_csv()`|Read CSV file|
||`df.to_csv()`|Write CSV file|

---

## Related Notes

- [[NumPy Reference]]
- [[Data Visualization with Matplotlib]]
- [[Statistical Analysis]]
- [[Machine Learning Preprocessing]]

#pandas #python #dataanalysis #datascience