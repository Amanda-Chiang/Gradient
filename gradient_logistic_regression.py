# Amanda Chiang, CSC561

from gradients_project import Variable
from sklearn.metrics import accuracy_score
import numpy as np

class LogisticRegression:

    def __init__(self, learn_rate = 0.01, grad_cutoff = 0.01):
        self.learn_rate = learn_rate
        self.grad_cutoff = grad_cutoff

        # arbitrary starter m and b values
        self.m_ideal =  0.1
        self.b_ideal =  0.2 

        # creating Variables which serve as placeholders for m_ideal and b_ideal when linked in a dictionary
        m_1 = Variable(1)
        b_1 = Variable(2)

        # learned the lambda function from Sarah Pan. Using it to store the function that finds the y_hat value
        self.y_hat = lambda x: 1 / (1 + Variable.exp(-1 * (m_1 * x + b_1)))

    # x_data and y_data are each numpy arrays of data
    def fit(self, X, y):

        # use this default value to ensure that regardless of the user's grad_cutoff, it still starts gradient descent
        grad_sum = 1 + self.grad_cutoff

        c = None

        # keep doing gradient descent til the absolute value of the sum of all components of the gradient is very tiny
        # (indicating very small change in cost, reaching the local minimum)
        while abs(grad_sum) > self.grad_cutoff:

            grad_sum = 1 + self.grad_cutoff

            c = self.cost(X, y)
            # If interested in spectating the cost
            # print("Cost: " + str(c) + ", m: " + str(self.m_ideal) + ", b: " + str(self.b_ideal))

            # find the gradient of the cost function with the current m and b values
            g2 = self.get_grad(X, y)  

            # re-calculating the gradient sum for this specific m and b set
            grad_sum = (g2[0]**2 + g2[1]**2)**0.5

            # update m and b values according to gradient and learn_rate
            self.m_ideal -= g2[0] * self.learn_rate
            self.b_ideal -= g2[1] * self.learn_rate

    # calculate the gradients of specific m and b values across a dataset
    def get_grad (self, X, y):
        # array for m and b gradients
        grad = np.array([0, 0])

        # adding the gradient of the loss of each x with the given m and b values
        for i in range(len(X)):
            # I wanted to implement what you told me about making cost a Variable to make my code faster, but I was running into bugs and ran out of time
            if y[i] == 0:
                grad = grad - Variable.log(1 - self.y_hat(X[i])).grad({'x_1': self.m_ideal, 'x_2': self.b_ideal})
            elif y[i] == 1:
                grad = grad - Variable.log(self.y_hat(X[i])).grad({'x_1': self.m_ideal, 'x_2': self.b_ideal})
            
        return grad / len(X)
    
    def cost (self, X, y):
        cost = 0
        for i in range(len(X)):
            # plugging in the cost formula. Using if statements as the selectors
            y_pred = self.y_hat(X[i]).evaluate({"x_1": self.m_ideal, "x_2": self.b_ideal})
            if y[i] == 0:                
                cost -= np.log(1 - y_pred)
            elif y[i] == 1:
                cost -= np.log(y_pred)
        return cost
    
    def predict(self, x_test):
        # p will be an array of y predictions for each x, utilizing the estimated m and b values
        p = []
        for x in x_test:
            p.append(self.y_hat(x).evaluate({"x_1": self.m_ideal, "x_2": self.b_ideal}))
        return p

def main():
    # testing on a data set with negative slope
    X1,y1 = [1, 2, 3, 4, 5, 6, 7, 8, 8, 10],[1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
    X_test1,y_true1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    l1 = LogisticRegression(0.01)
    l1.fit(X1,y1)
    y_preds1 = l1.predict(X_test1)

    print("Accuracy score, Negative slope: " + str(accuracy_score(y_true1, np.round(y_preds1))))

    # testing on a data set with positive slope
    X2,y2 = [1, 2, 3, 4, 5, 6, 7, 8, 8, 10],[0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    X_test2,y_true2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    l2 = LogisticRegression(0.01)
    l2.fit(X2,y2)
    y_preds2 = l2.predict(X_test2)

    print("Accuracy score, Positive slope: " + str(accuracy_score(y_true2, np.round(y_preds2))))

    # Another test with slightly positive slope
    X3,y3 = [0, 2, 3, 4, 5, 10, 32, 33, 35], [0, 0, 0, 0, 1, 1, 1, 1, 1]
    X_test3,y_true3 = [-2,1,6,12,15,20], [0, 0, 1, 1, 0, 1]

    l3 = LogisticRegression(0.01)
    l3.fit(X3,y3)
    y_preds3 = l3.predict(X_test3)

    print("Accuracy score: " + str(accuracy_score(y_true3, np.round(y_preds3))))

    # Test where training data is same as testing data to ensure that the algorithm is training
    X4, y4 = [1, 2, 3, 4, 5], [0, 0, 0, 1, 1]
    X_test4, y_true4 = [1, 2, 3, 4, 5], [0, 0, 0, 1, 1]

    l4 = LogisticRegression(0.01)
    l4.fit(X4,y4)
    y_preds4 = l4.predict(X_test4)

    # Accuracy score should be 1, and it is
    print("Accuracy score when test data = train data: " + str(accuracy_score(y_true4, np.round(y_preds4))))

if __name__ == "__main__":
    main()