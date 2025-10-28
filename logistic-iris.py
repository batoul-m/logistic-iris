import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent_logistic(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    W = np.zeros(n + 1)  # thetas
    X_b = np.c_[np.ones((m, 1)), X]  # x0 is set to 1 and concatenated with x

    for epoch in range(epochs):
        # y_pred = X_b.dot(W)   # (Linear Regression)
        z = X_b.dot(W)  # (Logistic Regression)
        y_pred = sigmoid(z)

        error = y_pred - y

        gradient = (1/m) * X_b.T.dot(error)   # .T is transport .dot is dot product

        W -= learning_rate * gradient

        if epoch % 100 == 0:
            # loss = (1/(2*m)) * np.sum(error**2)   # (Linear Regression)
            loss = -(1/m) * np.sum(y*np.log(y_pred + 1e-9) + (1 - y)*np.log(1 - y_pred + 1e-9))  # Logistic Loss
            print(f"Epoch {epoch}, Loss: {loss}")

    return W

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[:100, :2]  # Take first 100 samples (two classes: Setosa & Versicolor), first 2 features
    y = iris.target[:100]

    y = (y == 1).astype(int)

    # Train model
    weights = gradient_descent_logistic(X, y, learning_rate=0.1, epochs=1000)

    print("Intercept (w0):", weights[0])
    print("Coefficients (w1, w2):", weights[1:])

    plt.scatter(X[y==0][:,0], X[y==0][:,1], color='blue', label='Class 0')
    plt.scatter(X[y==1][:,0], X[y==1][:,1], color='red', label='Class 1')

    x1_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    x2_vals = -(weights[0] + weights[1]*x1_vals) / weights[2]
    plt.plot(x1_vals, x2_vals, color='green', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression on Iris Dataset')
    plt.legend()
    plt.show()

