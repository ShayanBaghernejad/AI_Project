# AI Project
# Code written by Shayan Baghernejad -> std num: 9935688
# ---------------------------------------------------------------------------------------------------------------
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pre
import math

# ---------------------------------------------------------------------------------------------------------------

# Load the CSV file, skipping the first row
data = pd.read_csv("/Users/shayansmacbook/Downloads/diabetes2.csv")
#nData = data.drop(['Pregnancies', 'SkinThickness'], axis=1)
# nData['Outcome'] = data['Outcome'].apply(lambda x: 1 if x == 1 else 0)
#data['Glucose'].replace(to_replace = 0,  method='ffill', inplace=True)
x = np.asarray(data)
pre = pre.StandardScaler()
x = pre.fit_transform(x[:,:-1])
y = np.asarray(data['Outcome'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)


# ---------------------------------------------------------------------------------------------------------------
# Logistic Regression Functions:
#################################################################################################################
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))

    return g


# ---------------------------------------------------------------------------------------------------------------

def compute_cost(X, y, w, b, lambda_):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      total_cost : (scalar) cost
    """

    m, n = X.shape

    loss_sum = 0

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j] * X[i][j]
            z_wb += z_wb_ij

        z_wb += b

        f_wb = sigmoid(z_wb)

        loss = (-y[i] * np.log(f_wb)) - ((1 - y[i]) * np.log(1 - f_wb))

        loss_sum += loss

    total_cost = (1 / m) * loss_sum

    return total_cost


# ---------------------------------------------------------------------------------------------------------------

def compute_gradient(X, y, w, b, lambda_):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j] * X[i][j]
            z_wb += z_wb_ij
        z_wb += b
        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i

        for j in range(n):
            dj_dw_ij = (f_wb - y[i]) * X[i][j]
            dj_dw[j] += dj_dw_ij

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


# ---------------------------------------------------------------------------------------------------------------

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant

    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing

#################################################################################################################
# ---------------------------------------------------------------------------------------------------------------

def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    # Loop over each example
    for i in range(m):
        z_wb = 0
        # Loop over each feature
        for j in range(n):
            # Add the corresponding term to z_wb
            z_wb_ij = w[j] * X[i][j]
            z_wb += z_wb_ij

        # Add bias term
        z_wb += b

        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)
        # Apply the threshold
        p[i] = (f_wb >= 0.5).astype(int)
    #print(p)
    return p


# ---------------------------------------------------------------------------------------------------------------

# Some gradient descent settings
np.random.seed(1)
#initial_w = 0.01 * (np.random.rand(8) - 0.5)
initial_w = np.zeros(x_train.shape[1])
initial_b = 0
iterations = 1000
alpha = 0.1
# ---------------------------------------------------------------------------------------------------------------

w, b, J_history, _ = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha,iterations, 0)
p = predict(x_test, w, b)
print('Test Accuracy: %f' % (np.mean(p == y_test) * 100))


# ---------------------------------------------------------------------------------------------------------------

def predict_diabetes():
    try:
        pregnancies = int(entry_pregnancies.get())
        glucose = int(entry_glucose.get())
        blood_pressure = int(entry_blood_pressure.get())
        skin_thickness = int(entry_skin_thickness.get())
        insulin = int(entry_insulin.get())
        bmi = float(entry_bmi.get())
        diabetes_pedigree_function = float(entry_diabetes_pedigree.get())
        age = int(entry_age.get())

        newData = np.array(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        newData = pre.transform(newData)
        y_pred = predict(newData, w, b)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if y_pred == 1:
            messagebox.showinfo("Prediction Result",
                                "Bad news, based on the information given, it looks like you have diabetes.")
        else:
            messagebox.showinfo("Prediction Result",
                                "Good news, according to the information given,it looks like you dont have diabetes.")

    except ValueError:
        # print(y_pred)
        messagebox.showerror("Error", "Please enter valid input values.")


# ---------------------------------------------------------------------------------------------------------------

# Create GUI window
window = tk.Tk()
window.title("Diabetes Prediction")
window.geometry("400x450")

# Create input labels and entry fields
label_pregnancies = tk.Label(window, text="Number of pregnancies:")
label_pregnancies.pack()
entry_pregnancies = tk.Entry(window)
entry_pregnancies.pack()

label_glucose = tk.Label(window, text="Glucose amount:")
label_glucose.pack()
entry_glucose = tk.Entry(window)
entry_glucose.pack()

label_blood_pressure = tk.Label(window, text="Blood Pressure:")
label_blood_pressure.pack()
entry_blood_pressure = tk.Entry(window)
entry_blood_pressure.pack()

label_skin_thickness = tk.Label(window, text="Skin Thickness:")
label_skin_thickness.pack()
entry_skin_thickness = tk.Entry(window)
entry_skin_thickness.pack()

label_insulin = tk.Label(window, text="Insulin:")
label_insulin.pack()
entry_insulin = tk.Entry(window)
entry_insulin.pack()

label_bmi = tk.Label(window, text="BMI:")
label_bmi.pack()
entry_bmi = tk.Entry(window)
entry_bmi.pack()

label_diabetes_pedigree = tk.Label(window, text="Diabetes Pedigree Function:")
label_diabetes_pedigree.pack()
entry_diabetes_pedigree = tk.Entry(window)
entry_diabetes_pedigree.pack()

label_age = tk.Label(window, text="Age:")
label_age.pack()
entry_age = tk.Entry(window)
entry_age.pack()

# Create predict button
predict_button = tk.Button(window, text="Predict", command=predict_diabetes)
predict_button.pack()

# Start GUI event loop
window.mainloop()
# ---------------------------------------------------------------------------------------------------------------
