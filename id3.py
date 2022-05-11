import math
import random
import re

import pandas as pd
import numpy as np

import treenode

def entropy(dataframe, classColumn):
    '''
    This function calculates the entropy of a dataframe with respect to the specified classColumn

    Parameters:
        dataframe - A pandas dataframe that has a column called classColumn
        classColumn - The column whose values you want to use for the entropy calculation
    Returns:
        A number containing the calculated entropy
    '''

    valueCounts = dataframe[classColumn].value_counts()
    totalSize = len(dataframe)
    entropySum = 0
    for count in valueCounts:
        p_i = count / totalSize
        entropySum += p_i * math.log2(p_i)

    return -entropySum

def informationGain(dataframe, attributeName, classColumn):
    '''
    This function calculates the informationGain (in terms of entropy) from splitting a dataframe on a particular attribute.

    Parameters:
        dataframe - A pandas dataframe containing the data to calculate information gain on
            It should have columns named with the specified column
        attributeName - The name of the column that you want to split the dataframe on
        classColumn - The name of the column that specifies the class value for this decision tree
    Returns:
        A number containing the calculated information gain
    '''

    initalEntropy = entropy(dataframe, classColumn)

    valueCounts = dataframe[attributeName].value_counts()
    for rowName in valueCounts.index:
        D_a_j = dataframe.loc[dataframe[attributeName] == rowName ]
        initalEntropy -= (len(D_a_j) / len(dataframe)) * entropy(D_a_j, classColumn)

    return initalEntropy

def ID3(dataframe, attributeColumns, classColumn):
    '''
    Run the ID3 decision tree training algorithm on a dataset.

    (No privacy measures are taken here.)

    Parameters:
        dataframe - A pandas dataframe containing the data to train on
        attributeColumns - The names of the columns in the dataframe that you want to use for the construction of the decision tree.
            These columns should all contain categorical data only
        classColumn - The name of the column that specifies the class value for this decision tree
    Returns:
        A TreeNode object containing a decision tree that classifies rows in the data
    '''

    # Base cases:
    # If no attributes are left, return a leaf node with the majority class value in the dataframe.
    valueCounts = dataframe[classColumn].value_counts()
    if len(attributeColumns) == 0:
        # Determine the majority class in the dataframe.
        majorityClass = valueCounts.idxmax()
        # The True in the tuple here indicates that it is a leaf node.
        return treenode.TreeNode((True, majorityClass))

    # Next, if all data in the dataframe has the same class value, return a leaf node with the remaining class value.
    if len(valueCounts) == 1:
        onlyRemainingClass = valueCounts.index[0]
        # The True in the tuple here indicates that it is a leaf node.
        return treenode.TreeNode((True, onlyRemainingClass))

    # Recursive step:
    # Determine the best attribute to classify D

    # This dictionary maps from column name to information gain
    informationGainForColumn = {}
    for attributeName in attributeColumns:
        informationGainForColumn[attributeName] = informationGain(dataframe, attributeName, classColumn)

    # Choose the attribute that gives the maximum information gain.
    bestAttribute = max(informationGainForColumn, key=informationGainForColumn.get)

    # The False in the tuple here indicates that it is not a leaf node.
    node = treenode.TreeNode((False, bestAttribute))

    attributeColumnsWithoutBestAttribute = [ attribute for attribute in attributeColumns if attribute != bestAttribute ]
    # For all values of the best attribute
    bestAttributeValueCounts = dataframe[bestAttribute].value_counts()
    for attributeValue in bestAttributeValueCounts.index:
        D_a_j = dataframe.loc[dataframe[bestAttribute] == attributeValue ]
        node.children[attributeValue] = ID3(D_a_j, attributeColumnsWithoutBestAttribute, classColumn)

    return node


def DP_ID3(dataframe, attributeMap, classColumn, classes, maxDepth, epsilon, currentDepth):
    '''
    Run the differentially private ID3 decision tree training algorithm on a dataset.

    Parameters:
        dataframe - A pandas dataframe containing the data to train on
        attributeMap - A dictionary containing all of the columns you want to use in the decision tree as keys
            Their values should be a list of all of the values of each attribute
            So these attributes should contain categorical data only
        classColumn - The name of the column that specifies the class value for this decision tree
        classes - The set of possible values in the classColumn
        maxDepth - The maximum depth that the tree can grow to
        epsilon - The privacy budget for the entire training
    Returns:
        A TreeNode object containing a decision tree that classifies rows in the data
    '''
    #divides the privacy budget
    epsilon1 = epsilon/(2*(maxDepth+1))

    #Determines the length of the dataframe and add noise
    numRows = len(dataframe) + np.random.laplace(0, (1/epsilon1))

    # Find the max number of values an attribute has
    maxAttributeValues = 0
    for attributeName, attributeValues in attributeMap.items():
        if len(attributeValues) > maxAttributeValues:
            maxAttributeValues = len(attributeValues)

    #Keeps track of whether the maximum depth or number of attributes has been exceeded or
    #if there are too few instances in a class
    outOfAttributes = len(attributeMap) == 0
    tooFewPeoplePerClasss = numRows/(maxAttributeValues*len(classes)) < math.sqrt(2)/epsilon1
    depthExceeded = maxDepth == currentDepth

    if outOfAttributes or tooFewPeoplePerClasss or depthExceeded:
        maxClass = ("",-10000000000)
        #Counts the number of occurences of each class and adds noise
        for classValue in classes:
            lenClass = len(dataframe.loc[dataframe[classColumn] == classValue]) + np.random.laplace(0, (1/epsilon1))
            if lenClass > maxClass[1]:
                maxClass = (classValue, lenClass)
        # The True in the tuple here indicates that it is a leaf node.
        return treenode.TreeNode((True, maxClass[0]))

    else:
        #further divides the privacy budget
        epsilon2 = epsilon1/(2*len(attributeMap))
        V_aDict = {}
        # Find the attribute that results in the "best" split of the data.
        for attributeName, attributeValues in attributeMap.items():
            V_aDict[attributeName] = 0
            for attributeValue in attributeValues:
                numAttrValueOccurrences = len(dataframe.loc[dataframe[attributeName] == attributeValue]) + np.random.laplace(0, (1/epsilon2))
                for classValue in classes:
                    numAttrValueOccurrencesWithClass = len(dataframe.loc[(dataframe[attributeName] == attributeValue ) \
                    & (dataframe[classColumn] == classValue)]) + np.random.laplace(0, (1/epsilon2))
                    # If the number of occurences is 0 or negative, a log could not be calculated with them,
                    # so we just do nothing by default.
                    if numAttrValueOccurrences > 0 and numAttrValueOccurrencesWithClass> 0:
                        V_aDict[attributeName] += numAttrValueOccurrencesWithClass * math.log2(numAttrValueOccurrencesWithClass/numAttrValueOccurrences)
        bestAttribute = max(V_aDict, key=V_aDict.get)

        # Create a new internal node indicating the tree will split on that bestAttribute.
        # The False in the tuple here indicates that it is not a leaf node.
        node = treenode.TreeNode((False, bestAttribute))

        attributeColumnsWithoutBestAttribute = {}
        for key, value in attributeMap.items():
            if key != bestAttribute:
                attributeColumnsWithoutBestAttribute[key] = value

        #For all values of the best attribute
        bestAttributeValueCounts = dataframe[bestAttribute].value_counts()
        for attributeValue in bestAttributeValueCounts.index:
            D_a_j = dataframe.loc[dataframe[bestAttribute] == attributeValue ]
            node.children[attributeValue] = DP_ID3(D_a_j, attributeColumnsWithoutBestAttribute, classColumn, classes, maxDepth, epsilon, currentDepth+1)

        return node

def printNoises(prefix, trueValue, epsilon):
    '''
    Displays the amount of noise any individual function call generated by giving a percent error.
    Parameters:
        prefix- An annotation that gets printed out before generating the amont of noise
        trueValue- The actual value of the count of values in the dataset.
        epsilon- Our privacy budget
    Returns:
        A printed statement depicting the total percentage error obtained by the noise generated
    '''
    noises = []
    for i in range(100):
        noises.append(np.random.laplace(0, (1/epsilon)))
    if trueValue != 0:
        percentErrors = [ abs(noise)/trueValue for noise in noises ]
        averagePercentError = 100 * np.mean(percentErrors)
    else:
        averagePercentError = 0
    print(f"{prefix} trueValue: {trueValue} epsilon: {epsilon:.4f} averagePercentError: {averagePercentError:.4f}")

# == layerFunctions ==
# The three functions below determine the privacy budget for each layer, and all take the following parameters:
#     m - the scaling parameter
#     n - the currentDepth in the tree, where the root node is at a depth of 0
#     d - the maxDepth the tree can grow to, plus 1

def boundedExponential(m, n, d):
    #splits the privacy budget by a factor of m for each layer, increasing as we go down the tree
    # b(n) = ((m-1)(m^n))/((m^d)-1)
    return ((m-1)*math.pow(m,n))/(math.pow(m,d)-1)

def reversedBoundedExponential(m, n, d):
    #splits the privacy budget by a factor of m for each layer, decreasing as we go down the tree
    reversedN = d - n - 1
    return ((m-1)*math.pow(m,reversedN))/(math.pow(m,d)-1)

def evenSplit(m, n, d):
    #splits the privacy buddget equally among each layer
    return 1 / d

class ID3Configuration:

    layerFunctions = {
        "boundedExponential": boundedExponential,
        "reversedBoundedExponential": reversedBoundedExponential,
        "evenSplit": evenSplit
    }

    def __init__(self):
        self.maxDepth = 4
        self.epsilon = 1
        self.mValue = 2
        self.aProportion = 0.5
        self.layerFunction = evenSplit

    def __init__(self, commandLineArgs):
        self.maxDepth = int(self.processArg(commandLineArgs.maxDepth))
        self.epsilon = self.processArg(commandLineArgs.epsilon)
        self.mValue = self.processArg(commandLineArgs.mValue)
        self.aProportion = self.processArg(commandLineArgs.aProportion)
        self.layerFunction = ID3Configuration.layerFunctions[commandLineArgs.layerFunction]

    def processArg(self, arg):
        #takes a string in the command line of the form "rand(a,b)" to generate a random value between a and b for a given parameter
        if arg.startswith("rand"):
            randCaptureRegex = r"rand\(([\d.]*),\s*([\d.]*)\)"
            randMin, randMax = re.findall(randCaptureRegex, arg)[0]
            return(random.uniform(float(randMin), float(randMax)))
        else:
            return float(arg)


def better_DP_ID3(dataframe, attributeMap, classColumn, classes, currentDepth, config: ID3Configuration):
    '''
    Run the differentially private ID3 decision tree training algorithm on a dataset.

    Parameters:
        dataframe - A pandas dataframe containing the data to train on
        attributeMap - A dictionary containing all of the columns you want to use in the decision tree as keys
            Their values should be a list of all of the values of each attribute
            So these attributes should contain categorical data only
        classColumn - The name of the column that specifies the class value for this decision tree
        classes - The set of possible values in the classColumn
        currentDepth - The current depth in the tree, with the root node at layer 0.  This should always be 0 when external code is calling this function.
        config - An ID3Configuration object that contains the numerical parameters that adjust how this function works.
    Returns:
        A TreeNode object containing a decision tree that classifies rows in the data
    '''

    # m is the scaling parameter
    m = config.mValue
    n = currentDepth
    d = config.maxDepth+1
    epsilonThisLayer = config.layerFunction(m, n, d) * config.epsilon
    epsilon1 = epsilonThisLayer * config.aProportion
    epsilonRest = epsilonThisLayer - epsilon1

    # Determine the length of the dataframe and add noise
    #printNoises("A: ", len(dataframe), epsilon1)
    numRows = len(dataframe) + np.random.laplace(0, (1/epsilon1))

    # Find the max number of values the remaining attributes have
    maxAttributeValues = 0
    for attributeName, attributeValues in attributeMap.items():
        if len(attributeValues) > maxAttributeValues:
            maxAttributeValues = len(attributeValues)

    #Keeps track of whether the maximum depth or number of attributes has been exceeded or
    #if there are too few instances in a class
    outOfAttributes = len(attributeMap) == 0
    tooFewPeoplePerClasss = numRows/(maxAttributeValues*len(classes)) < math.sqrt(2)/epsilon1
    depthExceeded = config.maxDepth == currentDepth

    if outOfAttributes or tooFewPeoplePerClasss or depthExceeded:
        maxClass = ("",-10000000000)
        #Counts the number of occurences of each class and adds noise
        for classValue in classes:
            #printNoises("D: ", len(dataframe.loc[dataframe[classColumn] == classValue]), epsilon1)
            lenClass = len(dataframe.loc[dataframe[classColumn] == classValue]) + np.random.laplace(0, (1/epsilonRest))
            if lenClass > maxClass[1]:
                maxClass = (classValue, lenClass)
        # The True in the tuple here indicates that it is a leaf node.
        return treenode.TreeNode((True, maxClass[0]))

    else:
        #further divides the privacy budget equally for each attribute
        epsilon2 = epsilonRest/(2*len(attributeMap))
        V_aDict = {}
        # Find the attribute that results in the "best" split of the data.
        for attributeName, attributeValues in attributeMap.items():
            V_aDict[attributeName] = 0
            for attributeValue in attributeValues:
                #printNoises("B: ", len(dataframe.loc[dataframe[attributeName] == attributeValue]), epsilon2)
                numAttrValueOccurrences = len(dataframe.loc[dataframe[attributeName] == attributeValue]) + np.random.laplace(0, (1/epsilon2))
                for classValue in classes:
                    #printNoises("C: ", len(dataframe.loc[(dataframe[attributeName] == attributeValue ) & (dataframe[classColumn] == classValue)]), epsilon2)
                    numAttrValueOccurrencesWithClass = len(dataframe.loc[(dataframe[attributeName] == attributeValue ) \
                    & (dataframe[classColumn] == classValue)]) + np.random.laplace(0, (1/epsilon2))
                    # If the number of occurences is 0 or negative, a log could not be calculated with them,
                    # so we just do nothing by default.
                    if numAttrValueOccurrences > 0 and numAttrValueOccurrencesWithClass> 0:
                        V_aDict[attributeName] += numAttrValueOccurrencesWithClass * math.log2(numAttrValueOccurrencesWithClass/numAttrValueOccurrences)
        bestAttribute = max(V_aDict, key=V_aDict.get)

        # Create a new internal node indicating the tree will split on that bestAttribute.
        # The False in the tuple here indicates that it is not a leaf node.
        node = treenode.TreeNode((False, bestAttribute))

        attributeColumnsWithoutBestAttribute = {}
        for key, value in attributeMap.items():
            if key != bestAttribute:
                attributeColumnsWithoutBestAttribute[key] = value

        #For all values of the best attribute
        bestAttributeValueCounts = dataframe[bestAttribute].value_counts()
        for attributeValue in bestAttributeValueCounts.index:
            D_a_j = dataframe.loc[dataframe[bestAttribute] == attributeValue ]
            node.children[attributeValue] = better_DP_ID3(D_a_j, attributeColumnsWithoutBestAttribute, classColumn, classes, currentDepth+1, config)

        return node

