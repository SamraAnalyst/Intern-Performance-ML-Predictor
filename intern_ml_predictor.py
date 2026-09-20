import pandas as pd
import numpy as np

performance_data = {
    "Intern_Name": ["Samra", "Ahmed", "Sana", "Zain", "Ali"],
    "Attendance_(%)": [95, 60, 88, 45, 90],
    "Task_Completed": [8, 3, 7, 2, 8],
    "Feedback_Score": [9, 4, 8, 3, 9],
    "Success_Probability": [1, 0, 1, 0, 1]

}

df = pd.DataFrame(performance_data)
print("---- Intern Analytical Performance Data Logs ----")
print(df)

from sklearn.linear_model import LogisticRegression
print("\n--- Step 1: Seperating Features (x) and Target (Y) ---")
X = df[["Attendance_(%)", "Task_Completed", "Feedback_Score"]]

Y = df["Success_Probability"]

print("\n--- Step 2: Training the Logistic Regression Model ---")
model = LogisticRegression()
model.fit(X, Y)
print("Model training is successfully complete!")

print("\n--- Step 3: Making an Expert Performance Prediction ---")
new_intern_data = np.array([[85, 6, 7]])

prediction = model.predict(new_intern_data)

print("Predicted Class Output for the new intern inut parameter: {prediction}")
if prediction[0] == 1:
    print("Result Analysis: This intern is highly likely to be SUCCESSFUL!")
else:
    print("Result Analysis: This intern NEEDS PERSONALIZED GUIDANCE!")
          
