"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
import submission
from submission import *
trainExamples = readExamples('polarity.train')
validationExamples = readExamples('polarity.dev')
featureExtractor = submission.extractWordFeatures
# although this movie suffers from some choice , this film is still worth watching


def main():
	weights = submission.learnPredictor(trainExamples, validationExamples, featureExtractor, 40, 0.01)
	interactivePrompt(featureExtractor, weights)


if __name__ == '__main__':
	main()