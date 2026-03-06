import pyspark, nltk, re, math
from pyspark.sql import SparkSession
import pyspark.sql.functions as sparkFunc
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
        exportList.append(((input[0], word), 1))
        # exportList.append((f"{input[0]}@{word}", 1))
        # exportList.append((input[0],word, 1))
    return exportList

def wordCount(input):
    output = len(input[1])
    return (input[0], output)

    







# ---------------------------------------------------
# processing summaryfile
# ---------------------------------------------------
#setting up nltk downloads
nltk.download('stopwords')
nltk.download('punkt')


spark = SparkSession.builder.appName("dataFrames").getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("OFF")
print("-----------------------------------------------------------------")
print("Begin actual Debug")



inFile = sc.textFile(f"./{summaryFile}")
splitNumRdd = inFile.map(splitNumAndSummary)
# calculate N for tf-idf
totalNumOfMovies = splitNumRdd.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).count()
wordCountPerMovie = splitNumRdd.map(wordCount)
mappedSplitRdd = splitNumRdd.flatMap(termOccourance)
termFrequencyPerMovie = mappedSplitRdd.reduceByKey(lambda x, y: x + y)

#calculating the nomalized TF by ((movie, word), normalizedTf)
joinPrepRdd = termFrequencyPerMovie.map(lambda x: (x[0][0],(x[0][1], x[1]) ))
preNormalizedTf = joinPrepRdd.join(wordCountPerMovie)
normalizedTfByMovie = preNormalizedTf.map(lambda x: ((x[0], x[1][0][0]), x[1][0][1] / x[1][1]))

#calculating document frequency
dfByTerm = normalizedTfByMovie.map(lambda x: (x[0][1], 1)).reduceByKey(lambda x, y: x + y)
idfByTerm = dfByTerm.map(lambda x: (x[0], math.log(totalNumOfMovies/x[1])))


#calculatin tf-idf ((movie, word), tf-idf)
tfByTerm = normalizedTfByMovie.map(lambda x: (x[0][1], (x[0][0], x[1])))
tfIdJoinfByTerm = tfByTerm.join(idfByTerm)
tfIdfByMovieandTerm = tfIdJoinfByTerm.map(lambda x: ((x[1][0][0], x[0]), x[1][0][1] * x[1][1]))




# print(tfIdfByTerm.take(10))




# ---------------------------------------------------
# processing metadata functions
# ---------------------------------------------------
def readMetadataLine(line):
    parts = line.split("\t")
    return (parts[0], parts[2])




# ---------------------------------------------------
# processing Metadata
# ---------------------------------------------------

metadataFile = sc.textFile("./movie.metadata.tsv")
metaDataPairs = metadataFile.map(readMetadataLine)
metaDataDict = dict(metaDataPairs.collect())





# ---------------------------------------------------
# processing searchFile
# ---------------------------------------------------

searchFile = open(f"./{searchTerms}")

for line in searchFile:
    line = line.lower()
    line = line.strip()
    words = line.split(" ")
    if(len(words) == 1):
        print(f"For the search term [{words[0]}] here are the top 10 movies:")
        arrOfMovies = tfIdfByMovieandTerm.filter(lambda x: x[0][1] == words[0]).sortBy(lambda x: x[1], ascending=False).take(10)

        for i, movie in enumerate(arrOfMovies, start=1):
            try:
                print(f"    {i}. {metaDataDict[movie[0][0]]}")
            except:
                print(f"    {i}. Movie title not in metadata Id is {movie[0][0]}")
    else:
        #removing stopwords from search query
        stopwrd= set(stopwords.words('english'))
        tokens = word_tokenize(line)
        filteredSearch = [word for word in tokens if word not in stopwrd]

        # calculating tfIdf of each of the search terms by multiplying its nromalized tf byt the idf of that search term
        totalWordsInSearch = len(filteredSearch)
        searchQuery = sc.parallelize(filteredSearch)
        searchWordNormalizedTf = searchQuery.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0], x[1]/totalWordsInSearch))
        tfIdfBySearchTerm = searchWordNormalizedTf.join(idfByTerm).map(lambda x: (x[0], x[1][0] * x[1][1]))
        
        #(term, (movie, tfidf)) calculating the dot product of search query tfidf with that of the corpus tfidf
        # then reducing this down into the sum of the tf-idf of each 
        tfidfbyTerm = tfIdfByMovieandTerm.map(lambda x: (x[0][1], (x[0][0], x[1])))
        joinedQueryTfidf = tfidfbyTerm.join(tfIdfBySearchTerm)
        dotProd = joinedQueryTfidf.map(lambda x: (x[1][0][0], x[1][0][1] * x[1][1]))
        dotByMovie = dotProd.reduceByKey(lambda a,b: a+b)

        #computing magnitude of the document vector
        documentMagnitude = tfIdfByMovieandTerm.map(lambda x: (x[0][0], math.pow(x[1],2))).reduceByKey(lambda a,b:a+b).mapValues(lambda x: math.sqrt(x))

        #computing magnitude of the search vector
        searchMagnitude = math.sqrt(sum(math.pow(v,2) for x,v in tfIdfBySearchTerm.collect()))

        #computing cosine similarity
        joinedMovies = dotByMovie.join(documentMagnitude)
        cosineSimilarityByMovie = joinedMovies.mapValues(lambda x: x[0] / (x[1] * searchMagnitude))
        results = cosineSimilarityByMovie.sortBy(lambda x: x[1], ascending=False)
        

        print(f"for the search terms {filteredSearch} here are the top 10 results:")
        for i, movie in enumerate(results.take(10), start=1):
            try:
                print(f"    {i}. {metaDataDict[movie[0]]}")
            except:
                print(f"    {i}. Movie title not in metadata Id is: {movie[0]}")
        


        



