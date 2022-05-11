import random

#This class creates a tree structure data type
class TreeNode:
    def __init__(self, value):
        self.children = {}
        self.data = value

    def __str__(self):
        output = str(self.data) + ":\n"
        for label, child in self.children.items():
            # Add an extra tab before each child line so they get indented properly.
            childContent = str(child).replace("\n\t", "\n\t\t")
            output += "\t" + label + ": " + childContent

        return output

    def evaluate(self, row):
        '''
        Use this decision tree to classify a row.

        Parameters:
            row - A pandas Series object containing a single row with data values for attributes that the tree can use
        Returns:
            The classification value output from the decision tree
        '''

        isLeaf, value = self.data
        if isLeaf:
            return value
        else:
            attributeValue = row[value]
            return self.children[attributeValue].evaluate(row)

class RandomGuesser:
    def __init__(self, dataframe, classColumn):
        self.guessFrequencies = {}

        valueCountsDict = dataframe[classColumn].value_counts().to_dict()
        totalSize = len(dataframe)
        for value, count in valueCountsDict.items():
            self.guessFrequencies[value] = count / totalSize

    def evaluate(self, row):
        # Generate a number between 0 and 1 to use for our guess.
        randomNumber = random.random()
        cumulativeTotal = 0
        for value, randChance in self.guessFrequencies.items():
            cumulativeTotal += randChance
            if (cumulativeTotal > randomNumber):
                return value
