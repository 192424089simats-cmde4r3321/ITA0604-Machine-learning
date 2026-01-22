# Candidate Elimination Algorithm

# Training data
# Attributes: Sky, AirTemp, Humidity, Wind, Water, Forecast
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High',   'Strong', 'Warm', 'Change', 'No']
]

# Initialize S and G
num_attributes = len(training_data[0]) - 1
S = ['Ø'] * num_attributes
G = [['?'] * num_attributes]

for example in training_data:
    x = example[:-1]
    label = example[-1]

    if label == 'Yes':   # Positive example
        for i in range(num_attributes):
            if S[i] == 'Ø':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

        # Remove inconsistent hypotheses from G
        G = [g for g in G if all(g[i] == '?' or g[i] == x[i] for i in range(num_attributes))]

    else:   # Negative example
        new_G = []
        for g in G:
            for i in range(num_attributes):
                if g[i] == '?' and S[i] != x[i]:
                    temp = g.copy()
                    temp[i] = S[i]
                    new_G.append(temp)
        G = new_G

# Output
print("Final Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypotheses (G):")
for g in G:
    print(g)
