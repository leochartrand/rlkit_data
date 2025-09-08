import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv('data/labels.csv')

df['Label'] = df['Label'].str.replace(' on ', ' ', regex=False)

# Display entire vocabulary
print("Vocabulary in the 'Label' column:")

# Remove floats from labels column
labels = df[~df['Label'].apply(lambda x: isinstance(x, float))]['Label']
uniques = sorted(labels.str.split().explode().unique())

print("Verbs:")
verbs = ["close", "open", "pickup", "put"]
for verb in verbs:
    if verb in uniques:
        uniques.remove(verb)
print(verbs)
print("Objects:")
print(uniques)

# Save back to CSV
df.to_csv('data/labels.csv', index=False)