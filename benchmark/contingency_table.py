#%%
import numpy as np
from tabulate import tabulate
from scipy.special import expit

#%%
min_group_01 = 1
max_group_01 = 5
elements_per_group = 3



#%%
def select_beta(elements_per_group,b):
    
    beta = np.zeros(elements_per_group*4)
    #possibilities = [7,8,-7,-8]
    for i in range(elements_per_group):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = np.random.uniform(b/7, b/5)
        else:
            value = np.random.uniform(-b/5,-b/7)
        beta[i] = value
    for i in range(elements_per_group*2,elements_per_group*3):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = np.random.uniform(b/7, b/5)
        else:
            value = np.random.uniform(-b/5,-b/7)
        beta[i] = value
    #beta[elements_per_group*4] = 20
    return beta

#%%

def toy_4group(elements_per_group, total_samples,z_prob,mean_1,mean_2,beta):
    total_features = elements_per_group*4
    z1_size = int(total_samples * z_prob)
    z1 = np.ones(z1_size)
    z2 = np.zeros(total_samples-z1_size)
    z = np.concatenate((z1,z2))
    g1 = np.zeros((elements_per_group,total_samples))
    g2 = np.zeros((elements_per_group,total_samples))
    g3 = np.zeros((elements_per_group,total_samples))
    g4 = np.zeros((elements_per_group,total_samples))
    
    for i in range(elements_per_group):
        for j in range(total_samples):
            if z[j] == 1:
                g1[i][j] = np.random.normal(mean_1,4)
                g2[i][j] = np.random.normal(mean_1,4)
            else:
                g1[i][j] = np.random.normal(0,4)
                g2[i][j] = np.random.normal(0,4)
            
        g3[i] = np.random.normal(mean_2,4,total_samples)
        g4[i] = np.random.normal(mean_2,4,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    #x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)

    
    mu = np.matmul(x,beta)
    gama = expit(mu)
    signal_to_noise = np.var(np.matmul(x,beta))
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y,beta, signal_to_noise


#%%
'''
It generates the dataset
Parameters:
    number_of_samples: How many sample points to generate
    non_protected_ratio: The ratio of non_protected_attribute to total samples
'''
def get_dataset(number_of_samples,non_protected_ratio):
    s = 1.25
    beta = select_beta(elements_per_group,s)
    mean_1 = np.random.uniform(min_group_01,max_group_01)
    mean_2 = np.random.uniform(min_group_01,max_group_01)
    x, z, y, beta, stn = toy_4group(elements_per_group,number_of_samples,non_protected_ratio,mean_1,mean_2,beta)
    return x, y, z

#%%
'''
Calculates the Demographic parity
Parameters:
    data: The samples
    labels: Class labels of the samples
    prediction: Prediction of the samples
    protectedIndex: Index of the column of the protected attribute
    protectedValue" The value of protected attribute
Output:
    Returns demographic parity
'''
def DP(data, labels, prediction,protectedIndex, protectedValue):
    protectedClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] == protectedValue]   
    elseClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] != protectedValue]
    #p = sum(1 for (x,l) in protectedClass if l == 1)
    #q = sum(1 for (x,l) in elseClass  if l == 1)
    protectedProb = sum(1 for (x,l) in protectedClass if l == 1) / len(protectedClass) if len(protectedClass) != 0 else 0
    elseProb = sum(1 for (x,l) in elseClass  if l == 1) / len(elseClass) if len(elseClass) != 0 else 0
    return (elseProb - protectedProb)
#%%
'''
This function outputs the expected fairness, left1right0 bias, left0right1 bias, and class probabilities at the left and right node
Parameters:
    leftX: samples at the left node
    rightX:Samples at the right node
    lefty:Class labels of samples at left node
    righty:Class labels of samples at the right node
    protected_attribute: Index of the column of protected attribute
    protected_value: The value of the protected attribute
Output:
    Expected fairness
    Bias at the left
    Bias at the right
    Class probabilities at left and right

'''
def fairness(leftX,lefty,rightX,righty,protected_attribute,protected_val):
    #print("probabilistic")
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    if len(countLeft) == 2:
        left0, left1 = countLeft[0]/len(lefty), countLeft[1]/len(lefty)
    if len(countRight) == 2:
        right0, right1 = countRight[0]/len(righty), countRight[1]/len(righty)
    if len(countLeft) == 1:
        left0 = countLeft[0]/len(lefty) if valueLeft[0] == 0 else 0
        left1 = countLeft[0]/len(lefty) if valueLeft[0] == 1 else 0
    if len(countRight) == 1:
        right0 = countRight[0]/len(righty) if valueRight[0] == 0 else 0
        right1 = countRight[0]/len(righty) if valueRight[0] == 1 else 0
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    pred00 = np.concatenate((np.zeros(len(lefty)),np.zeros(len(righty))), axis = 0)
    pred01 = np.concatenate((np.zeros(len(lefty)),np.ones(len(righty))), axis = 0)
    pred10 = np.concatenate((np.ones(len(lefty)),np.zeros(len(righty))), axis = 0)
    pred11 = np.concatenate((np.ones(len(lefty)),np.ones(len(righty))), axis = 0)

    #fairness_score00 = DP(x,y,pred00,protected_attribute,protected_val)
    fairness_score01 = DP(x,y,pred01,protected_attribute,protected_val)
    #print("left 0 prob, right 1 prob", left0, right1)
    fairness_score10 = DP(x,y,pred10,protected_attribute,protected_val)
    #print("left 1 prob, right 0 prob", left1, right0)
    #fairness_score11 = DP(x,y,pred11,protected_attribute,protected_val)
    
    #print(fairness_score00, fairness_score01, fairness_score10, fairness_score11)
    fairness_score =  fairness_score01*right1 + fairness_score10*left1
    print(fairness_score01, right1, fairness_score10, left1)
    print(abs(fairness_score))
    #print(fairness_score01,left0,right1,fairness_score10,left1,right0)
    return 1 - abs(fairness_score),fairness_score01,fairness_score10, left0, left1, right0, right1
#%%
def fairness_deter(leftX,lefty,rightX,righty,protected_attribute,protected_val):
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    max_left = np.argmax(countLeft)
    max_right = np.argmax(countRight)
    pred_left = np.full(len(lefty),max_left)
    pred_right = np.full(len(righty),max_right)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    
    fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return 1 - fairness_score, max_left, max_right
#%%
'''
This function outputs the contingency table
Parameter:
    number_of_samples: takes the number of sample in the dataset
    left ratio: The ratio of samples going to the left to the total number of samples
    non_protected_ratio: The ratio of non-protected samples to the total number of samples
Output:
    Prints two contingency table for left = 1 and right = 1, respectively
'''
def contingency_table(number_of_samples, left_ratio,non_protected_ratio):
    
    total_features = 4*elements_per_group
    x, y, z = get_dataset(number_of_samples, non_protected_ratio)
    x_with_protected = np.concatenate((x,np.reshape(z,(-1,1))),axis=1)
    non_protected_size = int(number_of_samples * non_protected_ratio)
    protected_size = number_of_samples - non_protected_size

    left_count = int(number_of_samples*left_ratio)
    right_count = number_of_samples - left_count
    total_indeces = np.arange(len(y))
    left_indexs = np.random.choice(len(y),left_count,replace=False)
    right_indexs = [i for i in total_indeces if i not in left_indexs]
    leftX = x_with_protected[left_indexs]
    lefty = y[left_indexs]
    rightX = x_with_protected[right_indexs]
    righty = y[right_indexs]
    ######calculate left############
    left_z0y1 = sum([1 for x in leftX if x[total_features] == 0])
    left_z1y1 = left_count - left_z0y1
    left_z0y0 = protected_size - left_z0y1
    left_z1y0 = non_protected_size - left_z1y1
    ########calculate right############
    right_z0y1 = sum([1 for x in rightX if x[total_features] == 0])
    right_z1y1 = right_count - right_z0y1
    right_z0y0 = protected_size - right_z0y1
    right_z1y0 = non_protected_size - right_z1y1
    ########Calculate values########
    e_fairness, fairness01, fairness10, probleft0,probleft1, probright0, probright1 = fairness(leftX,lefty,rightX,righty,total_features,0)

    #############formulate table################
    table_left = [['Protected', 'y hat = 1', 'y hat = 0'], ['z', left_z0y1, left_z0y0], ['z^C', left_z1y1, left_z1y0]]
    table_right = [['Protected', 'y hat = 1', 'y hat = 0'], ['z', right_z0y1, right_z0y0], ['z^C', right_z1y1, right_z1y0]]
    deterministic_fairness, max_left, max_right = fairness_deter(leftX,lefty,rightX,righty,total_features,0)
    print("samples at left", left_count,"samples at right", right_count)
    print("When left is 1, right is 0")
    print("probability of left=1",probleft1,"probability of right=0",probright0)
    print(tabulate(table_left, headers='firstrow', tablefmt='grid'))
    print("DP in this case",fairness10)
    print("\n")
    print("When left is 0, right is 1")
    print("probability of left=0",probleft0,"probability of right=1",probright1)
    print(tabulate(table_right, headers='firstrow', tablefmt='grid'))
    print("DP in this case",fairness01)
    print("\n")
    print("Expected fairness", e_fairness)
    print("deterministic fairness", deterministic_fairness,"max left", max_left, "max right", max_right)
# %%
number_of_samples = 100
left_ratio = 0.2
non_protected_ratio = 0.7
contingency_table(number_of_samples,left_ratio,non_protected_ratio)


# %%
