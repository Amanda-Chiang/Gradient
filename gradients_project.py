# Amanda Chiang, CSC561

import math
import numpy as np
import torch
import random

class Variable:

    #if var_num = None, it's an operation node
    def __init__(self, var_num = None):
        if var_num != None and var_num < 1:
            raise ValueError("Invalid Number. Inputs must be integers starting from 1")
        self.var_num = var_num

    def evaluate(self,values):
        return values["x_" + str(self.var_num)]
    
    def grad(self, values):
        # take number of keys in values dictionary and make a np array with x_1, x_2, ... x_n where n is number of values in values
        arr = np.zeros(len(values))
        
        # although self.var_num starts at 1, index will still start 0
        arr[self.var_num - 1] = 1
        return arr

    def __add__(self, other):
        return AdditionVariable(self,other)
    
    def __radd__(self, other):
        return AdditionVariable(self, other)
    
    def __mul__(self, other):
        return MultiplicationVariable(self,other)
    
    def __rmul__(self, other):
        return MultiplicationVariable(self,other)
    
    def __sub__(self, other):
        return self + (-1 * other)
    
    def __rsub__(self,other):
        return (-1 * self) + other
    
    def __pow__(self, other):
        return PowVariable(self, other)
    
    def __truediv__(self, other):
        return self * (other**(-1))
    
    def __rtruediv__(self, other):
        return (self**(-1)) * other
    
    @staticmethod
    def log(input):
        return LogVariable(input)
    
    @staticmethod
    def exp(input):
        return ExpVariable(input)

class AdditionVariable(Variable):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, values):
        if isinstance(self.right, (float,int)):
            return self.left.evaluate(values) + self.right
        return self.left.evaluate(values) + self.right.evaluate(values)
    
    def grad(self, values):
        if isinstance(self.right, (float,int)):
            return self.left.grad(values)
        if isinstance(self.left, (float,int)):
            return self.right.grad(values)
        return self.left.grad(values) + self.right.grad(values)

class MultiplicationVariable(Variable):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, values):
        if isinstance(self.right, (float,int)):
            return self.left.evaluate(values) * self.right
        return self.left.evaluate(values) * self.right.evaluate(values)
    
    def grad(self, values):
        if isinstance(self.right, (float,int)):
            return self.left.grad(values) * self.right
        if isinstance(self.left, (float,int)):
            return self.right.grad(values) * self.left
        return self.left.evaluate(values) * self.right.grad(values) + self.left.grad(values) * self.right.evaluate(values)

class PowVariable(Variable):
    def __init__(self, left, right):
        self.left = left
        # self.right is the exponent, which is always a number
        self.right = right

    def evaluate(self, values):
        if self.left.evaluate(values) == 0 and self.right < 0:
            raise ValueError("Denominator cannot be 0")
        return self.left.evaluate(values) ** self.right
    
    def grad(self, values):
        return self.right * self.left.evaluate(values)**(self.right - 1) * self.left.grad(values)

class LogVariable(Variable):
    def __init__(self, input):
        self.input = input

    def evaluate(self, values):
        return math.log(self.input.evaluate(values))
    
    def grad(self, values):
        return self.input.grad(values) / self.input.evaluate(values)

class ExpVariable(Variable):
    def __init__(self, input):
        self.input = input

    def evaluate(self, values):
        return math.e**self.input.evaluate(values)
    
    def grad(self, values):
        return self.input.grad(values) * math.e**self.input.evaluate(values)

@staticmethod
def var_test():
    print("*** Variable Tests ***")

    values = {"x_1": 3, "x_2": 4, "x_3": 1, "x_4": -1, "x_5": 6}

    x_1 = Variable(1)
    x_2 = Variable(2)
    x_3 = Variable(3)
    x_4 = Variable(4)

    # Addition tests
    y1 = x_1 + x_2
    y2 = 3 + x_1 + x_4 + 1
    assert y1.evaluate(values) == 7
    assert y2.evaluate(values) == 6
    print("Addition tests passed")

    # Subtraction tests
    y3 = x_1 - x_2
    y4 = 1 - x_1 + 2 - x_3 - x_4
    assert y3.evaluate(values) == -1
    assert y4.evaluate(values) == 0
    print("Subtraction tests passed")

    # Multiplication tests
    y5 = x_1 * x_2 * x_3
    y6 = 2 * x_1
    y7 = x_2 * 3 * x_4
    assert y5.evaluate(values) == 12
    assert y6.evaluate(values) == 6
    assert y7.evaluate(values) == -12
    print("Multiplication tests passed")

    # Division tests
    y8 = x_3 / x_1
    y9 = 1 / x_2
    y10 = x_3 / 4 / x_4
    assert (y8.evaluate(values) - 0.3 < 0.04)
    assert y9.evaluate(values) == 0.25
    assert y10.evaluate(values) == -0.25
    # Correctly raises ValueError for Division by Zero
    '''
    y11 = x_1 / (x_2 - x_2)
    print(y11.evaluate(values))
    '''
    print("Division tests passed")

    # Power tests
    y12 = x_1**2
    y13 = x_2**(-1)
    y14 = x_3**0
    assert y12.evaluate(values) == 9
    assert y13.evaluate(values) == 0.25
    assert y14.evaluate(values) == 1
    print("Power tests passed")

    # Log tests
    y15 = Variable.log(x_3)
    y16 = Variable.log(x_2)
    assert y15.evaluate(values) == 0
    assert y16.evaluate(values) <= math.log(4) + 0.1 and y16.evaluate(values) >= math.log(4) - 0.1
    # Correctly causes a math domain error
    '''
    y17 = Variable.log(x_1 - x_1)
    print(y17.evaluate(values))
    '''
    print("Log tests passed")

    # Exp tests
    y18 = Variable.exp(x_2)
    y19 = Variable.exp(x_3 - x_3)
    y20 = Variable.exp(x_4)
    assert y18.evaluate(values) <= math.e**(4) + 0.1 and y18.evaluate(values) >= math.e**(4) - 0.1
    assert y19.evaluate(values) == 1
    assert y20.evaluate(values) <= math.e**(-1) + 0.1 and y20.evaluate(values) >= math.e**(-1) - 0.1
    print("Exp tests passed")

    z = Variable.exp(x_1 + x_2**2) + 3 * Variable.log(27 - x_1 * x_2 * x_3)
    z_true = math.e**(3 + 16) + 3 * math.log(27 - 3 * 4 * 1)
    assert z.evaluate(values) <= z_true + 0.1 and z.evaluate(values) >= z_true - 0.1

    print("All Variable Tests Passed! :)")

@staticmethod
def grad_test():
    # I used Sarah and Laerdon's suggestions for PyTorch and learned the format for the example test you provided from Laerdon
    # that I used to supplement my tests

    print("*** Gradient Tests ***")
    
    values = {"x_1": 3, "x_2": 1, "x_3": 7}
    x_1 = Variable(1)
    x_2 = Variable(2)
    x_3 = Variable(3)
    x_4 = Variable(4)
    x_5 = Variable(5)

    z = Variable.exp(x_1 + x_2**2) + 3 * Variable.log(27 - x_1 * x_2 * x_3)

    x_1t = torch.tensor(3., requires_grad=True)
    x_2t = torch.tensor(1., requires_grad=True)
    x_3t = torch.tensor(7., requires_grad=True)

    zt = torch.exp(x_1t + x_2t**2) + 3 * torch.log(27 - x_1t * x_2t * x_3t)
    zt.backward()

    # Check that torch's variables are very close to my values
    assert (z.grad(values)[0] >= x_1t.grad - 0.1 and z.grad(values)[0] <= x_1t.grad + 0.1), f'z.grad[0] failed, {z.grad[0]}'
    assert (z.grad(values)[1] >= x_2t.grad - 0.1 and z.grad(values)[1] <= x_2t.grad + 0.1), f'z.grad[1] failed, {z.grad[1]}'
    assert (z.grad(values)[2] >= x_3t.grad - 0.1 and z.grad(values)[2] <= x_3t.grad + 0.1), f'z.grad[1] failed, {z.grad[2]}'

    values1 = {"x_1": 3, "x_2": 4, "x_3": 1, "x_4": -1, "x_5": 6}

    g1 = x_1 / 2 + 3 * x_2
    print("Grad of x_1 / 2 + 3 * x_2: " + str(g1.grad(values1)))
    # correctly prints [0.5, 3, 0, 0, 0]

    g2 = Variable.log(x_2) + (x_1**3)/4 + Variable.exp(x_1 - x_2) + x_4 * x_5
    print("Grad of ln(x_2) + (x_1**3) / 4 + e^(x_1 - x_2) + x_4 * x_5: " + str(g2.grad(values1)))
    # correctly prints [7.11787944117, -0.11787944117, 0, 6, -1]

    g3 = x_4 * x_5 + 3 * x_5
    print("Grad of x_4 * x_5 + 3 * x_5: " + str(g3.grad(values1)))
    # correctly prints [0,0,0,6,2]

    g4 = 3 - x_1
    print("Grad of 3 - x_1: " + str(g4.grad(values1)))
    # should output -1 * x_1, or [-1, 0, 0, 0, 0]. Currently outputs [-1, -0, -0, -0, -0] which is equivalent

    g5 = x_1 - 3
    print("Grad of x_1 - 3: " + str(g5.grad(values1)))
    # correctly outputs [1, 0, 0, 0, 0]

    # Determine the gradient of the function at the point (x_1, x_2, x_3) = (3, 1, 7):
    print(z.grad({"x_1": 3, "x_2": 1, "x_3": 7}))
    # Correctly outputs array([ 51.09815003,  98.69630007,  -1.5       ])

def main():
    var_test()
    grad_test()

if __name__ == "__main__":
    main()