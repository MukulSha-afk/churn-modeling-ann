# churn-modeling-ann
A machine learning project that uses Artificial Neural Networks to predict customer churn based on historical customer data, helping businesses identify and retain high-risk customers.
Challenges include:

Handling imbalanced data
Identifying important features influencing churn
Building a model that generalizes well

🧠 Approach

We use an Artificial Neural Network (ANN) to model customer behavior and predict churn.

Steps followed:

Data Preprocessing
Handling missing values
Encoding categorical variables (Gender, Geography)
Feature scaling using StandardScaler
Feature Selection
Selected relevant features like:
Credit Score
Age
Balance
Tenure
Estimated Salary
Model Building
Built ANN using:
Input layer
Hidden layers (Dense + ReLU)
Output layer (Sigmoid for binary classification)
Training
Loss function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
Evaluation

🔁 Iterations

Several improvements were made during development:

Tried different numbers of hidden layers

Tuned neurons per layer (32, 64, 128)

Adjusted learning rate

Experimented with batch sizes and epochs

Tested different activation functions

Applied dropout to reduce overfitting


⚙️ Key Design Choices
✔ ANN over traditional models

Captures complex non-linear relationships
Performs better on structured data with interactions

✔ Sigmoid Activation (Output Layer)
Suitable for binary classification

✔ ReLU Activation (Hidden Layers)
Avoids vanishing gradient problem

✔ Standardization
Important for neural network convergence

✔ Dropout Layers
Helps prevent overfitting

⏱️ Daily Time Commitment

Task	Time Spent
Data preprocessing	1–2 hours
Model building	2 hours
Hyperparameter tuning	2–3 hours
Evaluation & debugging	1–2 hours
Documentation	1 hour


🛠️ Tech Stack

Python
TensorFlow / Keras
NumPy
Pandas
Scikit-learn
Matplotlib


📌 Future Improvements

Use advanced models (LSTM, XGBoost comparison)

Deploy model using Flask / FastAPI

Add explainability (SHAP / LIME)

Handle class imbalance using SMOT
Confusion Matrix

Accuracy Score
