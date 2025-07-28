
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

n = 2000
data = pd.DataFrame({
'Category': np.random.choice(['A', 'B', 'C', 'D'], size=n),
'Value1': np.random.normal(loc=50, scale=15, size=n),
'Value2': np.random.normal(loc=30, scale=10, size=n),
'Group': np.random.choice(['X', 'Y'], size=n),
'Score': np.random.randint(60, 100, size=n)
})

bar_data = data['Category'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(bar_data.index, bar_data.values, color='skyblue', edgecolor='black')
plt.title('Bar Chart - Category Counts')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
colors = sns.color_palette('pastel')
explode = [0.05] * len(bar_data)
plt.pie(bar_data, labels=bar_data.index, autopct='%1.1f%%', startangle=90,colors=colors, explode=explode, shadow=True)
plt.title('Pie Chart - Category Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.set_palette("coolwarm")
sns.boxplot(data=data, x='Category', y='Value2', linewidth=2.5, fliersize=6,width=0.5)
plt.title("Boxplot with Custom Aesthetics")
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(data['Value2'], bins=20, color='skyblue', edgecolor='gray', alpha=0.85, histtype='stepfilled')
plt.title("Histogram with Aesthetic Enhancements")
plt.xlabel("Value2")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Value1', y='Value2', hue='Category',style='Group',
palette='husl', s=100, alpha=0.8, edgecolor='black')
plt.title("Scatter Plot with Aesthetic Tweaks", fontsize=14, weight='bold')
plt.grid(True, linestyle=':', linewidth=0.7)
plt.tight_layout()
plt.show()

plt.style.use('default')
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data['Value1'], data['Value2'], c=data['Score'],cmap='viridis', s=80,
alpha=0.9, edgecolors='black')
plt.colorbar(scatter, label='Score')
plt.title('Colormap Scatter Plot (Score)')
plt.xlabel('Value1')
plt.ylabel('Value2')
plt.tight_layout()
plt.show()

plt.style.use('white_background')
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data['Value1'], data['Value2'], c=data['Score'],cmap='plasma', s=80,
alpha=0.9, edgecolors='white')
plt.colorbar(scatter, label='Score')
plt.title('Colormap Scatter with Dark Theme')
plt.xlabel('Value1')
plt.ylabel('Value2')
plt.tight_layout()
plt.show()

plt.style.use('default')
sns.set_style('whitegrid')


from scipy.integrate import quad
result , error = quad ( lambda x: x**2 , 0, 3)
print (" Integral :", result )
