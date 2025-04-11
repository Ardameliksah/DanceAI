import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
csv_path = "C:/Users/ameli/Desktop/DanceAI/output_videos/rightTurnVideo/output_foot_positions.csv"
df = pd.read_csv(csv_path)

# Filter time between 5 and 10 seconds
mask = (df['timestamp'] >= 5) & (df['timestamp'] <= 10)
df_filtered = df[mask]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df_filtered['timestamp'], df_filtered['left_heel_y'], label='Left Heel Y', marker='o')
plt.plot(df_filtered['timestamp'], df_filtered['right_heel_y'], label='Right Heel Y', marker='x')
plt.title('Heel Y Positions (5s to 10s)')
plt.xlabel('Time (s)')
plt.ylabel('Y Position (pixels)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
