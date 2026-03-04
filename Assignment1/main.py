import os, nltk, numpy, re
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
# Extracing named entities
# ---------------------------------------------------

count = 0
namedEnitityList = []

#nltk setup
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True) 

# begin actually processing hte file
openFile = open(f"./{fileName}", 'r')


#reads lines untill first line of book is reached
for line in openFile:
    if(count < 35):
        line = openFile.readline()
        count += 1
        continue

    line =  openFile.readline()
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
                namedEnitityList.append(entity[0])
    
    # if(True):
    #     print("--------------------------------------")
    #     # print(namedEntities)
    #     print(namedEnitityList)
    #     count += 1
    count += 1
    print(count)

#once extraction is complete we want too write to a csv to store the values
midpointFile = open("./midpoint.csv", 'w')

for entity in namedEnitityList:
    midpointFile.write(f"{entity}\n")

    