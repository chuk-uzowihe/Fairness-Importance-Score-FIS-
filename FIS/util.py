import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.special import expit
import copy


def eqop(data,label, prediction, protectedIndex, protectedValue):
    protected = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] == protectedValue and l==1)]   
    el = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] != protectedValue and l==1)]
    protected_negative = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] == protectedValue and l==0)] 
    el_negative = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] != protectedValue and l==0)]
    tp_protected = sum(1 for (x,l,p) in protected if l == p)
    
    tp_el = sum(1 for (x,l,p) in el if l == p)
    tn_protected = sum(1 for (x,l,p) in protected_negative if l == p)
    tn_el = sum(1 for (x,l,p) in el_negative if l == p)
   
   
    tpr_protected = tp_protected / len(protected) if len(protected) != 0 else 0
    tpr_el = tp_el / len(el) if len(el) != 0 else 0
    
    tnr_protected = tn_protected / len(protected_negative) if len(protected_negative)!= 0 else 0
    tnr_el = tn_el / len(el_negative) if len(el_negative)!= 0 else 0
    negative_rate = tnr_protected - tnr_el
    eqop = (tpr_el - tpr_protected)
    
    return (eqop)
# %%

def DP(data, labels, prediction,protectedIndex, protectedValue):
    #print("changed")
    protectedClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] == protectedValue]   
    elseClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] != protectedValue]
    p = sum(1 for (x,l) in protectedClass if l == 1)
    q = sum(1 for (x,l) in elseClass  if l == 1)
    protectedProb = sum(1 for (x,l) in protectedClass if l == 1) / len(protectedClass) if len(protectedClass) != 0 else 0
    elseProb = sum(1 for (x,l) in elseClass  if l == 1) / len(elseClass) if len(elseClass) != 0 else 0
    #print("protected class, non-protected class, protected positive, non-protected positive",len(protectedClass),len(elseClass),p,q)
    return (elseProb - protectedProb)
#%%
def gini(y):
    total_classes, count = np.unique(y, return_counts=True)
    probability = np.zeros(len(total_classes), dtype=float)
    n = len(y)
    for i in range(len(total_classes)):
        probability[i] = (count[i]/n)**2
    if n == 0:
        return 0.0
    gini = 1 - np.sum(probability)
    return gini

# %%
def ma(x, window):
        return np.convolve(x, np.ones(window), 'valid') / window

#%%
def draw_plot(x,y,dest,name):
    sns.set_context("talk")
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x, y)
    plt.xlabel("Feature")
    plt.ylabel(name)
    plt.savefig(dest)
    plt.show()

#%%
def fairness(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
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
    if fairness_metric == 1:
        #fairness_score00 = eqop(x,y,pred00,protected_attribute,protected_val)
        fairness_score01 = eqop(x,y,pred01,protected_attribute,protected_val)
        #print("left 0 prob, right 1 prob", left0, right1)
        fairness_score10 = eqop(x,y,pred10,protected_attribute,protected_val)
        #print("left 1 prob, right 0 prob", left1, right0)
        #fairness_score11 = eqop(x,y,pred11,protected_attribute,protected_val)
    else:
        #fairness_score00 = DP(x,y,pred00,protected_attribute,protected_val)
        fairness_score01 = DP(x,y,pred01,protected_attribute,protected_val)
        #print("left 0 prob, right 1 prob", left0, right1)
        fairness_score10 = DP(x,y,pred10,protected_attribute,protected_val)
        #print("left 1 prob, right 0 prob", left1, right0)
        #fairness_score11 = DP(x,y,pred11,protected_attribute,protected_val)
    
    #print(fairness_score00, fairness_score01, fairness_score10, fairness_score11)
    fairness_score =  fairness_score01*right1 + fairness_score10*left1
    return 1 - abs(fairness_score)


def fairness_regression(leftX,lefty,rightX,righty,protected_attribute,protected_val,alpha = 1):
    #print("probabilistic")
    
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    
    
    protectedClass = [l for (x,l) in zip(x, y) 
        if x[protected_attribute] == protected_val]
    elseClass = [l for (x,l) in zip(x, y) 
        if x[protected_attribute] != protected_val]
    if len(protectedClass) == 0:
        fairness = 0
    elif len(elseClass) == 0:
        fairness = 0
    else:
        prot_val = np.mean(protectedClass)
        el_val = np.mean(elseClass)
        fairness = math.exp(-alpha * abs(prot_val - el_val))
        
    return fairness

def fairness_multiclass(leftX,lefty,rightX,righty,protected_attribute,protected_val,alpha = 1):
    max = 0
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    num_classes = np.unique(y)
    total_protected = len([l for (x,l) in zip(x, y) 
        if x[protected_attribute] == protected_val])
    total_el = len([l for (x,l) in zip(x, y) 
        if x[protected_attribute] != protected_val])
    for i in range(len(num_classes)):
        left_pk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] == protected_val])
        right_pk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] == protected_val])
        left_nk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] != protected_val])
        right_nk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] != protected_val])
        prob_leftk = ((left_pk) + (left_nk))/ len(leftX)
        prob_rightk = ((right_pk) + (right_nk))/ len(rightX)
        if total_protected == 0:
            bias = 1
        elif total_el == 0:
            bias = 1
        else:
            bias_left = (left_pk / total_protected - left_nk/ total_el)*prob_leftk
            bias_right = (right_pk / total_protected - right_nk/ total_el)*prob_rightk
            bias = abs(bias_left + bias_right)
        if bias > max:
            max = bias
    return 1 - max
def print_tree(model_dtree):
    n_nodes = model_dtree.node_count
    children_left = model_dtree.children_left
    children_right = model_dtree.children_right
    feature = model_dtree.feature
    threshold = model_dtree.threshold
    impurity = model_dtree.impurity
    fair_score = model_dtree.fair_score
    samples = model_dtree.number_of_data_points
    z0 = model_dtree.countz0
    z1 = model_dtree.countz1
    pos_z0 = model_dtree.positivez0
    pos_z1 = model_dtree.positivez1
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}leafnode={node}\n{space} z0 {z0},z1 {z1}\n{space} pos_z0 {pos_z0},pos_z1 {pos_z1} \n{space}Fairness: {sample}".format(
                    space=node_depth[i] * "\t", node=i,z0 = z0[i],
                    z1=z1[i],
                    pos_z0 = pos_z0[i],
                    pos_z1 = pos_z1[i],
                    sample =fair_score[i]
                )
            )
        else:
            print(
                "{space}splitnode={node}\n{space} impurity {impurity}, \n{space} fairness {fairness} \n{space} samples {sample} \n{space} z0 {z0}, z1 {z1}\n{space} pos_z0 {pos_z0},pos_z1 {pos_z1}\n{space} split {left} if X[:,{feature}] <= {threshold} else {right}."
                .format(
                    space=node_depth[i] * "\t",
                    node=i,
                    impurity = impurity[i],
                    fairness = fair_score[i],
                    sample = samples[i],
                    z0 = z0[i],
                    z1=z1[i],
                    pos_z0 = pos_z0[i],
                    pos_z1 = pos_z1[i],
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )
# %%
def previous_fairness(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    #print("changed")
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

    
    max_left = np.argmax(countLeft)
    max_right = np.argmax(countRight)
    
    '''
    if max_left == max_right:
        if len(countLeft == 1):
            max_right = ~max_right
        elif len(countRight == 1):
            max_left = ~max_left
        else:
            left_0,left_1 = countLeft[0]/len(lefty) , countLeft[1]/len(lefty)
            right_0,right_1 = countRight[0]/len(righty) , countRight[1]/len(righty)
            #print("same", max_left,max_right)
            if max_left == 0:
                if left_0 > right_0:
                    max_right = 1
                else:
                    max_left = 1
            else:
                if left_1 > right_1:
                    max_right = 0
                else:
                    max_left = 0
            #print("now", max_left,max_right)

    '''
    

    pred_left = np.full(len(lefty),max_left)
    pred_right = np.full(len(righty),max_right)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    
    if fairness_metric == 1:
        fairness_score = eqop(x,y,Prediction,protected_attribute,protected_val)
    else:
        fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return fairness_score

def fairness2(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    #print("bernoulli")
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
    
    fairness_score = []
    for i in range(100):
        pred_left = np.zeros(len(lefty))        
        pred_right = np.zeros(len(righty))
        for i in range (len(lefty)):
            pred_left[i] = np.random.binomial(1, left1)
        for i in range (len(righty)):
            pred_right[i] = np.random.binomial(1, right1)
        
        x = np.concatenate((leftX,rightX),axis=0)
        y = np.concatenate((lefty,righty),axis = 0)
        Prediction = np.concatenate((pred_left,pred_right), axis = 0)
        if fairness_metric == 1:
            fairness_score.append(eqop(x,y,Prediction,protected_attribute,protected_val)) 
        else:
            fairness_score.append(DP(x,y,Prediction,protected_attribute,protected_val)) 
    print(np.mean(fairness_score))
    return np.mean(fairness_score)

def fairness_rndm(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    #print("bernoulli")
    left_count = len(lefty)
    right_count = len(righty)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    fairness_score = []
    for i in range(100):
        total_indeces = np.arange(len(y))
        left_indexs = np.random.choice(len(y),left_count,replace=False)
        right_indexs = [i for i in total_indeces if i not in left_indexs]
        leftX = x[left_indexs]
        lefty = y[left_indexs]
        rightX = x[right_indexs]
        righty = y[right_indexs]
        fairness_score.append(fairness_wrong(leftX, lefty, rightX, righty,protected_attribute,protected_val,fairness_metric)) 
    
    return 1 - np.mean(fairness_score)


def fairness_wrong(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    max_left = np.argmax(countLeft)
    max_right = np.argmax(countRight)
    if max_left == max_right:
        if len(countLeft == 1):
            max_right = ~max_right
        elif len(countRight == 1):
            max_left = ~max_left
        else:
            left_0,left_1 = countLeft[0]/len(lefty) , countLeft[1]/len(lefty)
            right_0,right_1 = countRight[0]/len(righty) , countRight[1]/len(righty)
            #print("same", max_left,max_right)
            if max_left == 0:
                if left_0 > right_0:
                    max_right = 1
                else:
                    max_left = 1
            else:
                if left_1 > right_1:
                    max_right = 0
                else:
                    max_left = 0
            #print("now", max_left,max_right)

    pred_left = np.full(len(lefty),max_left )
    pred_right = np.full(len(righty),max_right)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    if fairness_metric == 1:
        fairness_score = eqop(x,y,Prediction,protected_attribute,protected_val)
    else:
        fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return fairness_score


def fairness_deterministic(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    max_left = np.argmax(countLeft)
    max_right = np.argmax(countRight)
    pred_left = np.full(len(lefty),max_left)
    pred_right = np.full(len(righty),max_right)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    
    if fairness_metric == 1:
        fairness_score = eqop(x,y,Prediction,protected_attribute,protected_val)
    else:
        fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return 1 - fairness_score





