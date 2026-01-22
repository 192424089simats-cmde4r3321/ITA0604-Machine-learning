# FIND-S Algorithm Implementation

def find_s(training_data):
    # Initialize hypothesis with None
    hypothesis = [None] * (len(training_data[0]) - 1)

    for example in training_data:
        if example[-1] == "Yes":  # Only positive examples
            for i in range(len(hypothesis)):
                if hypothesis[i] is None:
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = "?"

    return hypothesis


# Training data
training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"]
]

# Run FIND-S
final_hypothesis = find_s(training_data)

print("Most Specific Hypothesis:")
print(final_hypothesis)
