import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Training dataset
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes']
}

df = pd.DataFrame(data)

# Encode categorical data
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Train ID3 Decision Tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Classify new sample
new_sample = [[0, 2, 0, 1]]   # Sunny, Mild, High, Strong
prediction = model.predict(new_sample)

print("Predicted Class:", prediction)
