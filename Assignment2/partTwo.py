import nltk, pyspark, re, math, subprocess
from pyspark.sql import SparkSession
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ---------------------------------------------------
# helper functions
# ---------------------------------------------------

def lineExtractor(item: list):
    
    itemOne = item[1].replace("\n", "")
    itemOne = re.sub(r'[^\w\s]', '', itemOne)
    itemOne = itemOne.lower()
    itemOne = itemOne.strip()
    itemOne = itemOne.encode("ascii", "ignore").decode()

    #perfomring stopword removal and sentence split
    stopwrd= set(stopwords.words('english'))
    tokens = word_tokenize(itemOne)
    itemOne = [word for word in tokens if word not in stopwrd]

    # performing stemming 
    stemmer = PorterStemmer()
    itemOne = [stemmer.stem(word) for word in itemOne]


    itemZero  = item[0].lower()

    return (itemZero, itemOne)

def giveFinalStats(confusionMatrix: dict):
    
    accuracy = 0
    accuracy = (confusionMatrix['TP'] + confusionMatrix['TN']) / (confusionMatrix['TP'] + confusionMatrix['TN'] + confusionMatrix['FP'] + confusionMatrix['FN'])

    precision = 0
    precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP'])
    
    recall = 0
    recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN'])

    f1Score = 0
    f1Score = 2 * ((precision * recall) / (precision + recall))

    print(f"Confusion Matrix: {confusionMatrix}")
    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1 Score:  {f1Score}")


#setting up nltk downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
subprocess.run("wget -O spamEmailDataPartial.csv http://localhost:5000")


spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("OFF")
print("-----------------------------------------------------------------")
print("Begin actual Debug")


# ---------------------------------------------------
# processing summaryfile
# ---------------------------------------------------
# dataset from https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification
summaryFile = "spamEmailDataPartial.csv"


inFile = sc.textFile(f"./{summaryFile}")
df = spark.read.option("header", "true").option("multiLine", "True").option("quote", "\"").option("escape", "\"").csv(f"./{summaryFile}")
rddStart = df.rdd.map(list)
# creates a rdd of the format (label, [word, word, word])
lineKeyPair = rddStart.map(lineExtractor)
trainSplitRdd, testSplitRdd = lineKeyPair.randomSplit((0.80, 0.20), 56)

# ---------------------------------------------------
# creating model class
# ---------------------------------------------------
class NaiveBayesSpamHam:
    def __init__(self):
        self.classCounts = {}
        self.wordCounts = {}
        self.totalWords = {}
        self.vocab = set()
        self.vocabSize = 0
    
    # accepts a rdd of the format (label, [word, word, word])
    def train(self, inputRdd):

        #this gets the count of each class by doing a simple reduce by key count
        self.classCounts = dict(inputRdd.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).collect())
        
        # we need to setup the Rdd in the ((label, word), 1) format to prepare to get the sum of all the words per class
        wordLabelPairsRdd = inputRdd.flatMap(lambda x: [((x[0], word), 1) for word in x[1]])

        # this takes the previous rdd and the nreduces by key to get the sum of each word per class
        self.wordCounts = dict(wordLabelPairsRdd.reduceByKey(lambda x, y: x + y).collect())

        # we need the total number of words per class this simply makes pairs of (label, countOfWordsInThatRow) then reduces by key
        self.totalWords = dict(inputRdd.flatMap(lambda x: [(x[0], len(x[1]))]).reduceByKey(lambda x, y: x + y).collect())

        # we need to produce a vocab for use later and alos to get the size of the vocab 
        self.vocab = set(inputRdd.flatMap(lambda x: x[1]).distinct().collect())
        self.vocabSize = len(self.vocab)

        # print("classCount")
        # print(self.classCount)
        # print("WordCount")
        # print(self.wordCounts)
        # print("totalWords")
        # print(self.totalWords)
        # print("vocabSize")
        # print(self.vocabSize)
    
    # we need to log prior and log likleyhood to avaoid underflow error
    def logPriorPorbability(self, label):
        totalDocs = self.classCounts['spam'] + self.classCounts['ham']
        return math.log(self.classCounts[label] / totalDocs)
    
    def logLikelihoodPerWord(self, word, label):
        wordCount = self.wordCounts.get((label, word), 0)
        # we need to apply laplace smoothing to prevent divide by 0 error for unknown words
        return math.log((wordCount + 1) / (self.totalWords[label] + self.vocabSize))

    # accepts a ssingle email of the format (label, [word, word, word])
    def evalSingleEmail(self, email):
        prediction = ''
        spamScore = self.logPriorPorbability('spam')
        hamScore = self.logPriorPorbability('ham')

        for word in email[1]:
            spamScore += self.logLikelihoodPerWord(word, 'spam')
            hamScore += self.logLikelihoodPerWord(word, 'ham')

        if(spamScore > hamScore):
            prediction = 'spam'
        else:
            prediction = 'ham'
        
        print(f"Spam Prior: {self.logPriorPorbability('spam')}")
        print(f"Ham Prior: {self.logPriorPorbability('ham')}")

        # we dont care too much about extracting the individual results just the FN TN FP TP results
        if(prediction == 'spam' and email[0] == 'spam'):
            return ('TP', 1)
        elif(prediction == 'ham' and email[0] == 'spam'):
            return ('FN', 1)
        elif(prediction == 'spam' and email[0] == 'ham'):
            return ('FP', 1)
        elif(prediction == 'ham' and email[0] == 'ham'):
            return ('TN', 1)
        
    # accepts a rdd of the format (label, [word, word, word])
    def evaluate(self, evalRdd: pyspark.RDD):

        emailPredictions =  evalRdd.map(self.evalSingleEmail)
        predictionDict = dict(emailPredictions.reduceByKey(lambda x, y: x + y).collect())
        return predictionDict





model= NaiveBayesSpamHam()
model.train(trainSplitRdd)
confusionMatrixDict = model.evaluate(testSplitRdd)
giveFinalStats(confusionMatrixDict)
