import re

import pandas as pd

def readMushroomColumnDefinitions(namesFilePath):
    """
    Reads the names file for the Mushroom dataset to get all of the attributes and attribute values in the dataset

    Parameters:
        namesFilePath - a path to the agaricus-lepiota.names file
    Returns:
        A dictionary mapping from column names to an array of possible values
    """

    attributeMap = {}
    with open(namesFilePath) as file:
        fileContents = file.read()
        attributeStartPos = fileContents.find("7. Attribute Information")
        attributeEndPos = fileContents.find("\n\n", attributeStartPos)
        attributeSectionOfFile = fileContents[attributeStartPos:attributeEndPos]

        # This regex captures the attribute name in the first capturing group and the attribute contents in the second capturing group.
        attributeCaptureRegex = r"\d+\. ([\w\-\?]+):\s*([a-zA-z=,\n \?]*)"
        attributeCaptures = re.findall(attributeCaptureRegex, attributeSectionOfFile)
        for attributeName, attributeValueString in attributeCaptures:
            attributeMap[attributeName] = []
            attributeValueRegex = r"\=(\w)*"
            attributeValues = re.findall(attributeValueRegex, attributeValueString)
            for value in attributeValues:
                attributeMap[attributeName].append(value)

    return attributeMap


def readNurseryColumnDefinitions(namesFilePath):
    """
    Reads the names file for the Nursery dataset to get all of the attributes and attribute values in the dataset

    Parameters:
        namesFilePath - a path to the agaricus-lepiota.names file
    Returns:
        A dictionary mapping from column names to an array of possible values
    """

    attributeMap = {}
    with open(namesFilePath) as file:
        fileContents = file.read()
        start = fileContents.find("7. Attribute Values:")
        attribSectionStart =  fileContents.find("\n\n", start)
        stop = fileContents.find("8. Missing Attribute Values: none")
        attribSection = fileContents[attribSectionStart:stop].split("\n")[2:-2]
        for attributeLine in attribSection:
            attributes = attributeLine.strip().split()
            attributeMap[attributes[0]] = attributes[1:]
    return attributeMap


class DatasetConfiguration:
    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.attributeMap = {}
        self.classColumn = ""
        self.classes = []

def loadMushroomDataset() -> DatasetConfiguration:
    """
    Loads the mushroom dataset and returns a DatasetConfiguration object.
    """
    config = DatasetConfiguration()
    config.attributeMap = readMushroomColumnDefinitions('data/Mushrooms/agaricus-lepiota.names')
    columnNames = ["Class"] + list(config.attributeMap.keys())
    config.dataframe = pd.read_csv('data/Mushrooms/agaricus-lepiota.data', dtype=str, names=columnNames)
    config.classColumn = "Class"
    config.classes = ["p", "e"]

    return config

def loadBreastCancerDataset() -> DatasetConfiguration:
    """
    Loads the breast cancer dataset and returns a DatasetConfiguration object.
    """
    config = DatasetConfiguration()
    config.attributeMap = {
        "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
        "menopause": ["lt40", "ge40", "premeno"],
        "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
                  "45-49", "50-54", "55-59"],
        "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26",
                 "27-29", "30-32", "33-35", "36-39"],
        "node-caps": ["yes", "no"],
        "deg-malig": ["1", "2", "3"],
        "breast": ["left", "right" ],
        "breast-quad": ["left-up", "left-low", "right-up", "right-low", "central"],
        "irradiat": ["yes", "no"],
    }
    columnNames = ["Class"] + list(config.attributeMap.keys())
    config.dataframe = pd.read_csv('data/Breast Cancer/breast-cancer.data', dtype=str, names=columnNames)
    config.classColumn = "Class"
    config.classes = ["no-recurrence-events", "recurrence-events"]

    return config

def loadNurseryDataset() -> DatasetConfiguration:
    """
    Loads the nursery dataset and returns a DatasetConfiguration object.
    """
    config = DatasetConfiguration()
    config.attributeMap = readNurseryColumnDefinitions('data/Nursery/nursery.names')
    columnNames = columnNames = list(config.attributeMap.keys()) + ["Class"]
    config.dataframe = pd.read_csv('data/Nursery/nursery.data', dtype=str, names=columnNames)
    config.classColumn = "Class"
    config.classes = ["not_recom","recommend","very_recom","priority","spec_prior"]


    return  config

choices = {
    'mushroom': loadMushroomDataset,
    'breastCancer': loadBreastCancerDataset,
    'nursery': loadNurseryDataset
}
