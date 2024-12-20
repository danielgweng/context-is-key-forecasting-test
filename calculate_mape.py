import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Read the predictions
df = pd.read_csv('benchmark_results/CC-GPT-4o/CC-GPT-4o/GMVPredictionTask/predictions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate MAPE
mape = np.mean(np.abs((df['actual'] - df['forecast_1']) / df['actual'])) * 100

print(f"MAPE: {mape:.2f}%")
print(f"\nMean Actual: {df['actual'].mean():,.0f}")
print(f"Mean Forecast: {df['forecast_1'].mean():,.0f}")
print(f"Number of predictions: {len(df)}")

# Create the plot
plt.figure(figsize=(15, 8))
plt.plot(df['timestamp'], df['actual'], label='Actual', alpha=0.7)
plt.plot(df['timestamp'], df['forecast_1'], label='Forecast', alpha=0.7)
plt.title('GMV: Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('GMV')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Format y-axis to millions
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('forecast_vs_actual.png')
print("\nPlot saved as forecast_vs_actual.png") 
