# CS 401 Final Project

Evan Lang, Nicholas Reichert, Tessy Udoh

This project is a differentially-private implementation of the ID3 decision tree training algorithm using a different method of allocating the privacy budget.


## Prerequisites

This project was tested with the following versions of dependencies:

- Python 3.9.1
- pandas 1.3.3

In the data folder, we have provided 3 working test datasets:
- Breast Cancer: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer
- Mushrooms: https://archive.ics.uci.edu/ml/datasets/mushroom
- Nursery: https://archive.ics.uci.edu/ml/datasets/nursery

Other datasets to test with can be found here: https://archive.ics.uci.edu/ml/index.php

## Running

The dataset should be provided in the form of a Pandas dataframe.  You can load your dataframe from any source.  Our example code loads the dataframe from a file in the CSV format.

You can run `main` with the following command:

```bash
python3 main.py

```
If no additional command line arguments are provided, it will run with our default parameters. To see a list of all available arguments, you can run the following command:

```bash
python3 main.py -h

```
These (optional) arguments include:
- Dataset (d) : This specifies which of our 3 datasets to use
- Number of iterations (n): Number of decision trees to create and test
- Output file (o): file name for the resulting CSV file
- maxDepth: This is the maximum depth of the decision tree
- epsilon: The privacy budget for the decision tree
- mValue: The scaling factor for our epsilon-budget function. Must be greater than 1.
- aProportion: The proportion of the privacy budget per layer for the first count query. Must be between 0 and 1.
- layerFunction: The function (boundedExponential, reversedBoundedExponential, evenSplit) to use which determines how much budget each layer gets


## Results

To generate our resulting decision trees, we ran `main` using the following configurations:

```
python main.py -d mushroom -n 1000 -o mushroomAProp.csv --maxDepth 3 --epsilon 0.5 --mValue 2 --aProportion "rand(0,1)" --layerFunction evenSplit
python main.py -d nursery -n 1000 -o nurseryAProp.csv --maxDepth 3 --epsilon 0.5 --mValue 2 --aProportion "rand(0,1)" --layerFunction evenSplit
python main.py -d mushroom -n 1000 -o mushroomBExp.csv --maxDepth 3 --epsilon 0.5 --mValue "rand(1.001,4)" --aProportion 0.5 --layerFunction boundedExponential
python main.py -d nursery -n 1000 -o nuseryBExp.csv --maxDepth 3 --epsilon 0.5 --mValue "rand(1.001,4)" --aProportion 0.5 --layerFunction boundedExponential
python main.py -d mushroom -n 1000 -o mushroomBRev.csv --maxDepth 3 --epsilon 0.5 --mValue "rand(1.001,10)" --aProportion 0.5 --layerFunction reversedBoundedExponential
python main.py -d nursery -n 1000 -o nuseryBRev.csv --maxDepth 3 --epsilon 0.5 --mValue "rand(1.001,10)" --aProportion 0.5 --layerFunction reversedBoundedExponential
```
