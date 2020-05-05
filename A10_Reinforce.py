import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# pGT is a 1x6 vector
pGT = torch.Tensor([1./12, 2./12, 3./12, 3./12, 2./12, 1./12])
# y is a 1000x1 vector of randomly chosen integers from [0,5] with sampling probs taken from pGT
y = torch.from_numpy(np.random.choice(list(range(6)), size=1000, p=pGT.numpy())).type(torch.int64).view(-1, 1)
#delta is a 1000x6 with a single 1 per row that corresponds to the value taken from y
delta = torch.zeros(y.numel(),6).scatter(1,y,torch.ones_like(y).float())

print("delta ",delta)
print("pGT: " , pGT)
print("y: " , y)
#maximum likelihood given dataset y encoded in delta
def MaxLik(delta):
    alpha = 1
    theta = torch.randn(6)
    print("theta ", theta)
    for iter in range(100):
        p_theta = torch.nn.Softmax(dim=0)(theta)
#        print("p_theta " , p_theta)
        g = torch.mean(p_theta-delta,0)
        theta = theta - alpha*g
        print("p_theta: ", p_theta)
        print("pGT: " , pGT)
        print("Diff: %f" % torch.norm(p_theta - pGT))
    
    return theta

theta = MaxLik(delta)
print("theta " , theta)

#reinforce with reward R
def Reinforce(R, theta=None):
    alpha = 1
    y = torch.from_numpy(np.random.choice(list(range(6)), size=1000, p=pGT.numpy())).type(torch.int64).view(-1, 1)
    delta = torch.zeros([1000,6])
    curReward = torch.zeros([1000,1])
    if theta is None:
        #theta is a random 6x1 vector with entries chosen from N(0,1)
        theta = torch.randn(6)
#    for iter in range(10000):
    for iter in range(1000):
        print("iter: ", iter)
        #current distribution
        p_theta = torch.nn.Softmax(dim=0)(theta)
        print("p_theta:", p_theta)
        
        #sample from current distribution and compute reward
        ##############################
        ## Sample from p_theta, find the assignment delta and compute the reward
        ## for each sample
        ## Dimensions: cPT (6); y (1000x1 -> 1000x1); delta (1000x6); curReward (1000x1)
        ############################## 
        y[iter] = torch.argmax(p_theta)
        delta[iter][y[iter]] = 1 
#R = pGT so is a 1x6 vector
#needs to be running
#answer should converge to y = {3,4}. highest reward
        curReward[iter] = R[y[iter]]

        #compute gradient and update
        g = torch.mean(curReward*(delta - p_theta),0)
        theta = theta + alpha*g
        print("Diff: %f" % torch.norm(p_theta - pGT))
        print(p_theta)

R = pGT
Reinforce(R, theta)
    
