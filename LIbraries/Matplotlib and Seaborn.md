# Matplotlib & Seaborn Complete Reference

## Table of Contents

- [[#Matplotlib Fundamentals]]
- [[#Plot Types & Examples]]
- [[#Customization & Styling]]
- [[#Subplots & Layouts]]
- [[#Advanced Matplotlib]]
- [[#Seaborn Introduction]]
- [[#Seaborn Plot Types]]
- [[#Statistical Plots]]
- [[#Categorical Plots]]
- [[#Matrix Plots]]
- [[#Multi-plot Grids]]
- [[#Styling & Themes]]
- [[#Best Practices]]
- [[#Cheat Sheet]]

---

## Matplotlib Fundamentals

### Basic Setup

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Basic plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```

### The Figure-Axes Hierarchy

![Matplotlib Anatomy](https://matplotlib.org/stable/_images/anatomy.png)

**Key Components:**

- **Figure** - The entire plot window/page
- **Axes** - The plot area (what you normally think of as "plot")
- **Axis** - The x/y axis lines, ticks, labels

### Two Interfaces

```python
# 1. MATLAB-style (pyplot interface)
plt.plot(x, y)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('My Plot')

# 2. Object-oriented interface (RECOMMENDED)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('My Plot')
```

**Why OO Interface?**

- More explicit and clear
- Better for complex plots
- Easier to customize
- Essential for subplots

---

## Plot Types & Examples

### Line Plots

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Multiple lines
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linestyle='--', color='red')
ax.plot(x, np.tan(x), label='tan(x)', alpha=0.7)

ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
```

**Line Styles:**

- `'-'` or `'solid'` - Solid line
- `'--'` or `'dashed'` - Dashed line
- `'-.'` or `'dashdot'` - Dash-dot line
- `':'` or `'dotted'` - Dotted line

### Scatter Plots

```python
# Basic scatter
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar()  # Add color scale
```

![Scatter Plot Example](https://matplotlib.org/stable/_images/sphx_glr_scatter_plot_001.png)

### Bar Plots

```python
# Vertical bars
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Vertical
ax1.bar(categories, values, color=['red', 'green', 'blue', 'orange'])
ax1.set_title('Vertical Bar Plot')

# Horizontal
ax2.barh(categories, values, color='skyblue')
ax2.set_title('Horizontal Bar Plot')
```

### Histograms

```python
# Generate data
data = np.random.normal(100, 15, 1000)

fig, ax = plt.subplots()
n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')

# Add normal curve
x = np.linspace(data.min(), data.max(), 100)
ax.plot(x, (1/np.sqrt(2*np.pi*15**2)) * np.exp(-0.5*((x-100)/15)**2), 
        'r-', linewidth=2, label='Normal Curve')
ax.legend()
```

### Box Plots

```python
# Multiple datasets
data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(90, 20, 200)
data3 = np.random.normal(80, 30, 200)

fig, ax = plt.subplots()
box_plot = ax.boxplot([data1, data2, data3], 
                      labels=['Dataset 1', 'Dataset 2', 'Dataset 3'],
                      patch_artist=True)

# Customize colors
colors = ['lightblue', 'lightgreen', 'pink']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
```

![Box Plot Example](https://matplotlib.org/stable/_images/sphx_glr_boxplot_demo_001.png)

### Heatmaps

```python
# Create data
data = np.random.rand(10, 12)

fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
im = ax.imshow(data, cmap='hot', interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Values', rotation=270, labelpad=15)

# Add labels
ax.set_xticks(np.arange(12))
ax.set_yticks(np.arange(10))
ax.set_xticklabels([f'Col {i}' for i in range(12)])
ax.set_yticklabels([f'Row {i}' for i in range(10)])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
```

---

## Customization & Styling

### Colors

```python
# Color specification methods
plt.plot(x, y, color='red')           # Named colors
plt.plot(x, y, color='#FF5733')       # Hex codes
plt.plot(x, y, color=(0.1, 0.2, 0.5)) # RGB tuples
plt.plot(x, y, color='C0')            # Default color cycle
plt.plot(x, y, c='r')                 # Single letter abbreviations
```

**Named Colors:** `red`, `blue`, `green`, `black`, `white`, `gray`, `orange`, `purple`, `brown`, `pink`, `olive`, `cyan`

### Markers

```python
# Common markers
markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', '+', 'x']

for i, marker in enumerate(markers):
    plt.plot(i, i, marker=marker, markersize=10, label=marker)

plt.legend(ncol=4)
```

![Matplotlib Markers](https://matplotlib.org/stable/_images/sphx_glr_marker_reference_001.png)

### Text and Annotations

```python
fig, ax = plt.subplots()
ax.plot(x, y)

# Add text
ax.text(5, 0.5, 'Peak here!', fontsize=12, ha='center')

# Add annotation with arrow
ax.annotate('Local maximum', 
            xy=(7.85, 0.99), xytext=(6, 1.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, ha='center')

# Mathematical expressions (LaTeX)
ax.set_title(r'$y = \sin(x)$', fontsize=16)
ax.set_xlabel(r'$x$ values in radians')
```

### Fonts and Sizes

```python
# Global font settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11
})
```

---

## Subplots & Layouts

### Basic Subplots

```python
# 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot in each subplot
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')

axes[0, 1].plot(x, np.cos(x), 'r--')
axes[0, 1].set_title('cos(x)')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('tan(x)')
axes[1, 0].set_ylim(-5, 5)

axes[1, 1].plot(x, np.exp(-x/10))
axes[1, 1].set_title('exp(-x/10)')

plt.tight_layout()  # Adjust spacing
```

### Complex Layouts with GridSpec

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)

# Large plot spanning multiple cells
ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
ax1.plot(x, np.sin(x))
ax1.set_title('Main Plot')

# Smaller plots
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x, np.cos(x))

ax3 = fig.add_subplot(gs[1, 1:])  # Second row, last two columns
ax3.scatter(np.random.randn(100), np.random.randn(100))

ax4 = fig.add_subplot(gs[2, :])
ax4.bar(['A', 'B', 'C'], [1, 2, 3])
```

### Subplot Sharing

```python
# Shared axes
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

ax1.plot(x, np.sin(x))
ax1.set_ylabel('sin(x)')

ax2.plot(x, np.cos(x))
ax2.set_ylabel('cos(x)')
ax2.set_xlabel('x values')

# Remove x-axis labels from top plot
ax1.tick_params(labelbottom=False)
```

---

## Advanced Matplotlib

### 3D Plots

```python
from mpl_toolkits.mplot3d import Axes3D

# 3D Surface plot
fig = plt.figure(figsize=(12, 5))

# Surface plot
ax1 = fig.add_subplot(121, projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('3D Surface')

# 3D Scatter
ax2 = fig.add_subplot(122, projection='3d')
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
ax2.scatter(x, y, z, c=z, cmap='viridis')
ax2.set_title('3D Scatter')
```

### Animations

```python
import matplotlib.animation as animation

fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(frame):
    line.set_ydata(np.sin(x + frame/10))
    return line,

ani = animation.FuncAnimation(fig, animate, frames=200, 
                            interval=50, blit=True, repeat=True)

# Save as GIF
# ani.save('animation.gif', writer='pillow', fps=20)
```

### Interactive Widgets

```python
import matplotlib.widgets as widgets

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0 * np.sin(2 * np.pi * f0 * t)
l, = plt.plot(t, s, lw=2)

# Add slider
ax_freq = plt.axes([0.25, 0.1, 0.50, 0.03])
s_freq = widgets.Slider(ax_freq, 'Freq', 0.1, 30.0, valinit=f0)

def update(val):
    freq = s_freq.val
    l.set_ydata(a0 * np.sin(2 * np.pi * freq * t))
    fig.canvas.draw_idle()

s_freq.on_changed(update)
```

---

## Seaborn Introduction

### Why Seaborn?

- **Built on Matplotlib** - All matplotlib functionality available
- **Statistical focus** - Designed for statistical visualization
- **Beautiful defaults** - Attractive plots with minimal code
- **DataFrame integration** - Works seamlessly with pandas
- **Complex plots simplified** - Multi-panel figures made easy

### Basic Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load example dataset
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
flights = sns.load_dataset("flights")
```

### Built-in Datasets

```python
# Available datasets
print(sns.get_dataset_names())

# Common ones:
# tips, iris, flights, titanic, car_crashes, diamonds
```

---

## Seaborn Plot Types

### Distribution Plots

#### Histograms and KDE

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
sns.histplot(data=tips, x="total_bill", ax=axes[0,0])

# KDE (Kernel Density Estimation)
sns.kdeplot(data=tips, x="total_bill", ax=axes[0,1])

# Combined
sns.histplot(data=tips, x="total_bill", kde=True, ax=axes[1,0])

# Multiple distributions
sns.histplot(data=tips, x="total_bill", hue="time", ax=axes[1,1])

plt.tight_layout()
```

![Seaborn Distribution Plots](https://seaborn.pydata.org/_images/distributions_8_0.png)

#### Rug and ECDF Plots

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Rug plot (shows individual data points)
sns.histplot(data=tips, x="total_bill", ax=ax1)
sns.rugplot(data=tips, x="total_bill", ax=ax1)

# ECDF (Empirical Cumulative Distribution Function)
sns.ecdfplot(data=tips, x="total_bill", ax=ax2)
```

### Relationship Plots

#### Scatter Plots

```python
# Basic scatter
sns.scatterplot(data=tips, x="total_bill", y="tip")

# With categorical variables
sns.scatterplot(data=tips, x="total_bill", y="tip", 
                hue="time", style="smoker", size="size")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
```

#### Line Plots

```python
# Time series data
flights_wide = flights.pivot(index="year", columns="month", values="passengers")

plt.figure(figsize=(12, 8))
sns.lineplot(data=flights, x="year", y="passengers", hue="month")
plt.title("Flight Passengers Over Time")
```

#### Regression Plots

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Simple regression
sns.regplot(data=tips, x="total_bill", y="tip", ax=axes[0])

# With categorical variable
sns.regplot(data=tips, x="size", y="tip", ax=axes[1])

# Residual plot
sns.residplot(data=tips, x="total_bill", y="tip", ax=axes[2])

plt.tight_layout()
```

---

## Statistical Plots

### Joint Plots

```python
# Scatter with marginal distributions
sns.jointplot(data=tips, x="total_bill", y="tip", kind="scatter")

# With regression line
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg")

# Hexbin for dense data
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex")

# KDE
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde")
```

![Seaborn Joint Plot](https://seaborn.pydata.org/_images/regression_65_0.png)

### Pair Plots

```python
# All pairwise relationships
sns.pairplot(iris)

# With categorical variable
sns.pairplot(iris, hue="species")

# Customize diagonal
sns.pairplot(iris, hue="species", diag_kind="kde")
```

![Seaborn Pair Plot](https://seaborn.pydata.org/_images/axis_grids_5_0.png)

---

## Categorical Plots

### Strip and Swarm Plots

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Strip plot (with jitter)
sns.stripplot(data=tips, x="day", y="total_bill", ax=axes[0])

# Swarm plot (non-overlapping)
sns.swarmplot(data=tips, x="day", y="total_bill", ax=axes[1])

# With hue
sns.swarmplot(data=tips, x="day", y="total_bill", hue="time", ax=axes[2])

plt.tight_layout()
```

### Box and Violin Plots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plot
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0,0])

# Violin plot
sns.violinplot(data=tips, x="day", y="total_bill", ax=axes[0,1])

# Box plot with hue
sns.boxplot(data=tips, x="day", y="total_bill", hue="time", ax=axes[1,0])

# Violin plot with split
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", 
               split=True, ax=axes[1,1])

plt.tight_layout()
```

### Bar Plots

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Count plot
sns.countplot(data=tips, x="day", ax=axes[0])

# Bar plot (with aggregation)
sns.barplot(data=tips, x="day", y="total_bill", ax=axes[1])

# With error bars and hue
sns.barplot(data=tips, x="day", y="total_bill", hue="time", 
            ci=95, ax=axes[2])

plt.tight_layout()
```

---

## Matrix Plots

### Heatmaps

```python
# Correlation matrix
corr_matrix = tips.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True,           # Show correlation values
            cmap='coolwarm',      # Color scheme
            center=0,             # Center colormap at 0
            square=True,          # Square cells
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix')
```

### Pivot Table Heatmap

```python
# Create pivot table
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")

plt.figure(figsize=(12, 8))
sns.heatmap(flights_pivot, 
            annot=True, 
            fmt="d",              # Integer format
            cmap="YlOrRd")
plt.title('Flight Passengers by Month and Year')
```

![Seaborn Heatmap](https://seaborn.pydata.org/_images/matrix_8_0.png)

### Clustermap

```python
# Hierarchical clustering
sns.clustermap(flights_pivot, 
               cmap="viridis",
               standard_scale=1,    # Standardize rows
               figsize=(12, 8))
```

---

## Multi-plot Grids

### FacetGrid

```python
# Create grid based on categorical variable
g = sns.FacetGrid(tips, col="time", row="smoker", margin_titles=True)
g.map(sns.scatterplot, "total_bill", "tip")
g.add_legend()

# With different plot types
g = sns.FacetGrid(tips, col="day", height=4, aspect=0.8)
g.map(sns.histplot, "total_bill", bins=20)
```

### PairGrid (More Control than PairPlot)

```python
g = sns.PairGrid(iris, hue="species")
g.map_diag(sns.histplot, kde=True)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.add_legend()
```

### JointGrid

```python
g = sns.JointGrid(data=tips, x="total_bill", y="tip")
g.plot(sns.scatterplot, sns.histplot)
g.plot_marginals(sns.rugplot, color="red", height=0.1)
```

---

## Styling & Themes

### Built-in Styles

```python
# Available styles
styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, style in enumerate(styles):
    sns.set_style(style)
    ax = axes[i]
    sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax)
    ax.set_title(f'Style: {style}')

plt.tight_layout()
```

### Color Palettes

```python
# Qualitative palettes
palettes = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, palette in enumerate(palettes):
    sns.set_palette(palette)
    ax = axes[i]
    sns.boxplot(data=tips, x="day", y="total_bill", ax=ax)
    ax.set_title(f'Palette: {palette}')

plt.tight_layout()
```

![Seaborn Color Palettes](https://seaborn.pydata.org/_images/color_palettes_8_0.png)

### Custom Palettes

```python
# Sequential palette
sns.set_palette(sns.color_palette("Blues", 8))

# Diverging palette
sns.set_palette(sns.diverging_palette(240, 10, n=9))

# Custom colors
custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
sns.set_palette(custom_palette)
```

### Context and Scale

```python
# Context affects size of elements
contexts = ["paper", "notebook", "talk", "poster"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for i, context in enumerate(contexts):
    sns.set_context(context)
    ax = axes[i]
    sns.lineplot(data=flights, x="year", y="passengers", ax=ax)
    ax.set_title(f'Context: {context}')

plt.tight_layout()
```

---

## Best Practices

### Figure Size and DPI

```python
# Set figure size and resolution
plt.figure(figsize=(12, 8), dpi=300)  # High DPI for publications

# Or with seaborn
sns.set_context("paper", font_scale=1.2, rc={"figure.figsize": (10, 6)})
```

### Saving Figures

```python
# High-quality output
plt.savefig('my_plot.png', 
            dpi=300,           # High resolution
            bbox_inches='tight', # Remove extra whitespace
            facecolor='white',   # White background
            edgecolor='none')

# Vector format for publications
plt.savefig('my_plot.pdf', bbox_inches='tight')
plt.savefig('my_plot.svg', bbox_inches='tight')
```

### Performance Tips

```python
# For large datasets
# 1. Sample data
large_data_sample = large_data.sample(n=10000)

# 2. Use rasterization for scatter plots
plt.scatter(x, y, rasterized=True)

# 3. Reduce number of bins in histograms
sns.histplot(data, bins=30)  # Instead of default 50+

# 4. Use hexbin for very dense scatter plots
plt.hexbin(x, y, gridsize=20)
```

### Accessibility

```python
# Colorblind-friendly palette
sns.set_palette("colorblind")

# High contrast
sns.set_palette("dark")

# Add patterns for additional distinction
hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
```

---

## Cheat Sheet

### Quick Plot Functions

|Plot Type|Matplotlib|Seaborn|
|---|---|---|
|**Line**|`plt.plot()`|`sns.lineplot()`|
|**Scatter**|`plt.scatter()`|`sns.scatterplot()`|
|**Bar**|`plt.bar()`|`sns.barplot()`|
|**Histogram**|`plt.hist()`|`sns.histplot()`|
|**Box**|`plt.boxplot()`|`sns.boxplot()`|
|**Heatmap**|`plt.imshow()`|`sns.heatmap()`|

### Essential Customizations

```python
# Matplotlib
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 5)

# Seaborn (use matplotlib for customization)
ax = sns.scatterplot(data=df, x='col1', y='col2')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label') 
ax.set_title('Title')
```

### Color Specifications

```python
# Named colors
'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'

# Hex colors
'#FF5733', '#33FF57', '#3357FF'

# RGB tuples
(0.2, 0.4, 0.6)

# Seaborn palettes
'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'
```

### Statistical Plot Quick Reference

```python
# Distribution
sns.histplot()      # Histogram
sns.kdeplot()       # Kernel density
sns.ecdfplot()      # Empirical CDF
sns.rugplot()       # Rug plot

# Categorical
sns.stripplot()     # Strip plot
sns.swarmplot()     # Swarm plot
sns.boxplot()       # Box plot
sns.violinplot()    # Violin plot
sns.barplot()       # Bar plot
sns.countplot()     # Count plot

# Relational
sns.scatterplot()   # Scatter plot
sns.lineplot()      # Line plot
sns.regplot()       # Regression plot

# Matrix
sns.heatmap()       # Heatmap
sns.clustermap()    # Clustered heatmap

# Multi-plot
sns.pairplot()      # Pair plot
sns.jointplot()     # Joint plot
sns.FacetGrid()     # Facet grid
```

### Common Styling

```python
# Figure setup
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("talk")

# Remove spines
sns.despine()

# Rotate labels
plt.xticks(rotation=45)

# Tight layout
plt.tight_layout()
```

---

## Integration Examples

### Matplotlib + Seaborn

```python
# Use seaborn for statistical plots, matplotlib for fine control
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Seaborn plots
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0,0])
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", ax=axes[0,1])
sns.histplot(data=tips, x="total_bill", ax=axes[1,0])

# Pure matplotlib
axes[1,1].pie(tips.groupby('day').size(), labels=tips['day'].unique(), autopct='%1.1f%%')

# Matplotlib customization
for ax in axes.flat:
    ax.tick_params(labelsize=10)

plt.suptitle('Restaurant Tips Analysis', fontsize=16, y=1.02)
plt.tight_layout()
```

### Pandas + Plotting

```python
# Direct from pandas
tips.groupby('day')['total_bill'].mean().plot(kind='bar')

# Or use plot accessor
tips.plot.scatter(x='total_bill', y='tip', c='size', 
                  colormap='viridis', figsize=(8, 6))
```

---

## Related Notes

- [[#Pandas Complete Reference Notes]]
- [[#Data Analysis Workflow]]
- [[#Statistical Analysis]]
- [[#Machine Learning Visualization]]
- [[#Publication-Ready Plots]]

#matplotlib #seaborn #python #datavisualization #datascience #plotting