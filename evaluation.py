import pandas as pd

def calculateAccuracyOfDecisionTree(decisionTree, dataframe, classColumn):
    '''
    Calculate the accuracy of a decision tree at classifying rows in a dataset.
    This function classifies each row in the dataframe and sees if the decision tree's output matches the true value.

    Parameters:
        decisionTree - A TreeNode object representing the root of a decision tree you want to use for classification
        dataframe - A dataframe of many rows that contains data in the format the decision tree expects
        classColumn - The column that contains the class value
    Returns:
        A number between 0 and 1 representing the proportion of rows that are correctly classified by the decision tree
    '''

    countCorrect = 0
    for index, row in dataframe.iterrows():
        decisionTreeResult = decisionTree.evaluate(row)
        if decisionTreeResult == row[classColumn]:
            countCorrect += 1
    return countCorrect / len(dataframe)

def calculateFScoreOfDecisionTree(decisionTree, dataframe, classColumn):
    '''
    Calculate the F1 score of a decision tree at classifying rows in a dataset.
    This function classifies each row in the dataframe and calculates an f score based on the decision tree's output.

    Parameters:
        decisionTree - A TreeNode object representing the root of a decision tree you want to use for classification
        dataframe - A dataframe of many rows that contains data in the format the decision tree expects
        classColumn - The column that contains the class value
    Returns:


    '''
    classes = pd.unique(dataframe[classColumn])
    trueClassCountDict = dataframe[classColumn].value_counts().to_dict()
    predictedClassCountDict = dict.fromkeys(classes, 0)
    correctPredictedClassDict = dict.fromkeys(classes, 0)
    for index, row in dataframe.iterrows():
        decisionTreeResult = decisionTree.evaluate(row)
        predictedClassCountDict[decisionTreeResult] += 1
        if decisionTreeResult == row[classColumn]:
            correctPredictedClassDict[decisionTreeResult] += 1

    fScoreDict = dict.fromkeys(classes,0)
    for c in classes:
        if predictedClassCountDict[c] != 0:
            precison = correctPredictedClassDict[c]/predictedClassCountDict[c]
        else:
            precison = 0
        recall = correctPredictedClassDict[c]/trueClassCountDict[c]
        if precison == 0 or recall == 0:
            fscore = 0
        else:
            fscore = 2 * (precison * recall)/ (precison + recall)
        fScoreDict[c] = fscore


    macroFScore = (sum(fScoreDict.values()))/len(fScoreDict)
    weightedFScore = 0
    for c in classes:
        weightedFScore += trueClassCountDict[c] * fScoreDict[c]
    weightedFScore = weightedFScore/len(dataframe)

    return macroFScore, weightedFScore
