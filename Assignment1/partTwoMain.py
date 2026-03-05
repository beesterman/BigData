import pyspark, nltk, re
from pyspark.sql import SparkSession
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

summaryFile = "plot_summaries.txt"
searchTerms = "searchTerms.txt"

# ---------------------------------------------------
# processing functions
# ---------------------------------------------------

def splitNumAndSummary(input: str):
    count = 0
    numEndInd = 0
    summaryBeginInd = 0


    input = input.lower()
    input = input.strip()
    input = re.sub(r'[^\w\s]', '', input)


    for letter in input:
        if(letter.isdigit()):
            count += 1 
            numEndInd = count
            continue
        elif(letter.isalpha()):
            summaryBeginInd = count
            # print(f"The first letter is: {letter}")
            # print(f"The movie num is: {input[:numEndInd]}")
            # print(f"The summry is: {input[summaryBeginInd:]}")
            break
            # return input[:numEndInd], input[summaryBeginInd:]
        else:
            count += 1
            continue


    moveiID = input[:numEndInd]
    reviewString = input[summaryBeginInd:]


    stopwrd= set(stopwords.words('english'))
    tokens = word_tokenize(reviewString)
    filteredReview = [word for word in tokens if word not in stopwrd]

    return (moveiID, filteredReview)


def termOccourance(input):
    exportList = []
    for word in input[1]:
        exportList.append((f"{input[0]}@{word}", 1))
        # exportList.append((f"{word}", 1))
    return exportList

# def wordCount(input):


    







# ---------------------------------------------------
# processing summaryfile
# ---------------------------------------------------
#setting up nltk downloads
nltk.download('stopwords')
nltk.download('punkt')



sc = pyspark.SparkContext(appName="myApp").getOrCreate()
sc.setLogLevel("OFF")
print("-----------------------------------------------------------------")
print("Begin actual Debug")


inFile = sc.textFile(f"./{summaryFile}")
splitNumRdd = inFile.map(splitNumAndSummary)
mappedSplitRdd = splitNumRdd.flatMap(termOccourance)
reducedWordCount = mappedSplitRdd.reduceByKey(lambda x, y: x + y)

print(reducedWordCount.take(10))





# ---------------------------------------------------
# processing searchFile
# ---------------------------------------------------

searchFile = open(f"./{searchTerms}")

for line in searchFile:
    line = line.lower()
    words = line.split(" ")