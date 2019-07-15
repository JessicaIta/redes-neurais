import numpy as np

#Função Step Function
def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

#Função Sigmoide
def sigmoidFunction(soma):
    return 1 / (1 - np.exp(-soma))

#Função Tangente Hiperbólica
def hiperbolicTanFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

#Função ReLU
def reluFunction(soma):
    if (soma >= 0):
        return soma
    return 0

#Função Linear
def linearFunction(soma):
    return soma

#Função Softmax
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

#Função Softplus
def softplusFunction(x):
    return np.log(np.exp(x) + 1)

#Função Softsign
def softSignFunction(x):
    return 1 / (np.abs(x) + 1)

