

from newsapi import NewsApiClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

#newsapi setup
newsapi = NewsApiClient(api_key='c0183918a0f649b6a78e98c495f9100b')



spark = SparkSession\
        .builder\
        .appName("StreamGen")\
        .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

def splitLine(input):
    return

x = 0
while(x != 1):

    # make call to api to extract info
    all_articles = newsapi.get_everything(q='goverment',
                                      language='en',
                                      sort_by='relevancy',
                                      page=1)
    inFile = open("./testfile.txt", "w")
    for i in all_articles:
        inFile.write(i)
    print(all_articles)
    print(len(all_articles))
    
    x += 1

