import numpy as np

#defining activation fucntions
def sigmoid(x, deriv = False):
    if deriv==True:
        return x(1-x)
    return 1/(1+np.exp(-x))

def ELU(x, a,  deriv == False):
    if x > 0:
        if deriv == True:
            return 1
        return x
    elif deriv == True:
        return x-a
    return a(np.exp(x)-1)

X = np.array([[1,0,0],
[0,1,1],
[1,0,1], 
[1,1,1]])

y = np.array([[0],[1],[1],[0]])   
)

#randomise weights
syn0 = 2*np.random.rand((3,10))-1
syn1 = 2*np.random.rand((10,10))-1
syn2 = 2*np.random.rand((10,1))-1

#The Net
for j in xrange(60000):

    l0 = X
    l1 = ELU(np.dot(syn0, l0))
    l2 = ELU(np.dot(syn1, l1))
    l3 = sigmoid(np.dot(syn2, l2))

    #backprop
    l3error = y - l3
    if j%1000 == 0:
        print 'Error' + str(np.mean(np.abs(l3error)))
    l3delta = l3error*sigmoid(l3, deriv=True)
    l2error = l3delta.dot(syn2.T)
    l2delta = l2error*ELU(l2, 1, deriv=True)
    l1error = l2delta.dot(syn1.T)
    l1delta = l1error*ELU(l1,1,deriv= True)

    syn0 += l1.T.dot(l1error)
    syn1 += l2.T.dot(l2error)
    syn2 += l3.T.dot(l3error)

print 'output after training'
print l3
