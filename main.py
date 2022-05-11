#===============================================================================
# Tessy Udoh, Evan Lang, Nicholas Reichert
# CS 401 - Final Course Project
# Differentially Private ID3
# 11 December 2021
#===============================================================================
import argparse

import pandas as pd

import datasets
import evaluation
import id3
import treenode


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="mushroom", choices=datasets.choices.keys(), help="the dataset to use")
    parser.add_argument("-n", "--numIterations", type=int, default=10, help="the number of decision trees to create and test")
    parser.add_argument("-o", "--outputFile", default="output.csv", help="the output file to write data to in CSV format")
    parser.add_argument("--maxDepth", default="4", help="the maximum depth the tree can grow to")
    parser.add_argument("--epsilon", default="1", help="the privacy budget to use for each tree")
    parser.add_argument("--mValue", default="2", help="the scaling factor to use in the layerFunction. Must be greater than 1")
    parser.add_argument("--aProportion", default="0.5", help="the proportion of the privacy budget at each layer to give to the first count query.  Must be between 0 and 1.")
    parser.add_argument("--layerFunction", default="evenSplit", choices=id3.ID3Configuration.layerFunctions.keys(), help="the function to use which determines how much budget each layer gets")
    args = parser.parse_args()

    datasetConfig = datasets.choices[args.dataset]()

    csvRows = []
    for i in range(args.numIterations):
        paramConfig = id3.ID3Configuration(args)
        model_better_dp_id3 = id3.better_DP_ID3(datasetConfig.dataframe, datasetConfig.attributeMap, datasetConfig.classColumn, datasetConfig.classes, 0, paramConfig)
        accuracy = evaluation.calculateAccuracyOfDecisionTree(model_better_dp_id3, datasetConfig.dataframe, datasetConfig.classColumn)
        macroF1Score, weightedF1Score = evaluation.calculateFScoreOfDecisionTree(model_better_dp_id3, datasetConfig.dataframe, datasetConfig.classColumn)
        csvRows.append([paramConfig.maxDepth, paramConfig.epsilon, paramConfig.mValue, paramConfig.aProportion, paramConfig.layerFunction.__name__, accuracy, macroF1Score, weightedF1Score])

        if ((i+1) % 10 == 0):
            print(f"Progress: {i+1}/{args.numIterations} iterations complete")

    columnNames = ["maxDepth", "epsilon", "mValue", "aProportion", "layerFunction", "accuracy", "macroF1Score", "weightedF1Score"]
    decisionTreeEvaluation = pd.DataFrame(csvRows, columns=columnNames)
    decisionTreeEvaluation.to_csv(args.outputFile)

main()
