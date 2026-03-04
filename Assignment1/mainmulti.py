import os, nltk, numpy, re, linecache, math, multiprocessing, threading
from nltk import word_tokenize, pos_tag, ne_chunk 


# ---------------------------------------------------
# retreiving files
# ---------------------------------------------------
# https://www.gutenberg.org/ebooks/2641

fileName = "pg2641.txt"
fileURL = "https://www.gutenberg.org/cache/epub/2641/pg2641.txt"
fileFound = False



#checks to see if file exists
dirContent = os.listdir()

for file in dirContent:
    file = file.lower()
    if(file == fileName):
        fileFound = True

if(not fileFound):
   os.system(f"wget {fileURL}")
else:
    print("Found File")


# ---------------------------------------------------
# Multithread setup
# ---------------------------------------------------

processes = []
numOfWorkers = 10
lineSize = 100
numOfLines = 0
numOfFullItterations = 0
numOfLinesRemaining = 0
itterationCount = 0

#is 26 to account for file header
currentLine = 26

#mutex for write file
writeMutex = threading.Lock()

#if you want to process a file
processing = False



#nltk setup
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True) 

#attaining number of lines
openFile = open(f"./{fileName}", 'r')
for line in openFile:
    numOfLines += 1
#account for closing and header from guttenburg
numOfLines = numOfLines - 351
numOfLines -= 26
openFile.close()

#calculating number of times it will have to itterate
numOfFullItterations = math.trunc(numOfLines/(lineSize * numOfWorkers))
numOfLinesRemaining = numOfLines - (numOfFullItterations * (lineSize * numOfWorkers))

# ---------------------------------------------------
# Worker Process
# ---------------------------------------------------

def workerThread(fileName, start, stop):

    namedEntitiesStore = []
    namedEntities = []

    for i in range(start, stop):
        try:
            line = linecache.getline(fileName, i)
        except:
            break


        #removing whitespaces and punctuation
        line = line.strip()
        line = re.sub(r'[^\w\s]', '', line)

        #performs part of speech tagging
        tokens = word_tokenize(line)
        posTags = pos_tag(tokens) 
        namedEntities = ne_chunk(posTags)

        #extracts named entities from the list
        if(namedEntities.leaves()):
            for entity in namedEntities.leaves():
                if(entity[1] == "NNP"):
                    namedEntitiesStore.append(entity[0])

    #aquire write mutex and write to output file
    with writeMutex:
        midpointFile = open("./midpointMulti.csv", 'a')
        
        for entity in namedEntitiesStore:
            midpointFile.write(f"{entity}\n")

        midpointFile.close()


    linecache.clearcache()

    print(f"completed lines {start + 26} to {stop- 1 + 26} \n")

# ---------------------------------------------------
# MultiProcessing disbatch
# ---------------------------------------------------
namedEnitityList = []
isComplete = False

if(processing):
    print(f"Begining {numOfFullItterations} itterations")
    while(itterationCount != numOfFullItterations):
        for i in range(0, numOfWorkers):
            startPage = 0
            stopPage = 0

            startPage = currentLine
            stopPage = currentLine + lineSize
            currentLine += lineSize

            processes.append(multiprocessing.Process(target=(workerThread), args=(fileName, startPage, stopPage)))

        for i in range(0, numOfWorkers):
            processes[i].start()

        for i in range(0, numOfWorkers):
            processes[i].join()

        processes.clear()
        itterationCount += 1

    print(f"Only {numOfLinesRemaining} remain")
    while(isComplete == False):
        if((currentLine + lineSize) > numOfLines):
            workerThread(fileName, currentLine, currentLine + (numOfLines - currentLine))
        else:
            startPage = currentLine
            stopPage = currentLine + lineSize
            currentLine += lineSize
            workerThread(fileName, startPage, stopPage)

