# Fake-News-detector-on-AWS
FAKE NEWS DETECTOR

ABSTRACT:
The rise in amount of fake news has been very evident since the digital technology and this rise has encouraged work in various areas. The impact of fake news is alarming as it can mislead people to an extent that it can cause panic. Thus, identifying fake news has become one of the popular studies in data science. We are using machine learning and big data technologies to help us achieve our goal to identify news as fake or real. These technologies are available to us in the form of services by world leading cloud service provider, Amazon Web Services (AWS). We use open-source data containing fake and real news articles that we are using to train, build and deploy a machine learning model. For faster results we are using parallel processing framework: Apache Spark. This framework is accessible through AWS Elastic Map   Reduce (EMR) service which is the cloud big data platform for processing vast amount of data. Amazon Sagemaker service provides us python API to support Spark. We save our model in AWS file system using Simple Storage Service (S3). Our web application is built with flask web framework technology. This application takes news article as input from user and uses the saved model to identify it as fake or real.    

Keywords: Machine Learning, Big Data, Amazon Web Services (AWS), Flask, Apache Spark. 

INTRODUCTION:
Big data is a revolution that has reached plethora of businesses. Ninety percent of the data has been generated in the last two years. Of all data right now, less than 0.5 % is ever used or analyzed. So, there is a lot of potential yet to discover. Major organizations have started investing millions of dollars in Big data projects. 

The increase in the information the organizations have about people has raised concerns due to its capability of manipulating an individual’s decision. Knowing an individual’s habits, likes and activities would give a powerful hold to the organization over an individual. This hold could have a destructive effect on a person’s life. One of such instances was the 2016 US presidential elections. The Great Hack documentary about the famous scandal between Facebook-Cambridge Analytica showed how computer technology and data analysis can bring out the dark side of social media by reshaping the world in a particular image. The political parties hired Cambridge Analytica to use 5000 data points about each of the people the company held to manipulate a certain set of people. As a social media user ourselves, it has become difficult what to believe. We use our knowledge and skills to build a web application which can differentiate between real and fake news using machine learning. 

Three major technologies involved in the making of the application are a processing framework to process the data, building a machine learning model and deploying it to the web application. 

Apache Spark: Big data Processing Framework
Apache Spark is a cluster computing framework which allows clusters to programmed with implicit data parallelism and fault tolerance. It is a fast engine to process vast amount of data. It provides sophisticated libraries for machine learning and data analytics. We use Python API of Spark (Pyspark). 

Pyspark.ml: Machine Learning API
Pyspark offers pyspark.ml package which is a dataframe based machine learning API. This package provides APIs that helps configure practical machine learning pipelines. We can build an end-to-end machine learning pipeline using this package.

Flask: Web Application Framework
Flask is a web framework written in python. It is used to build a web application using the tools and libraries provided by it. It is easy to use and is highly customizable.

DATASET: 
We are using an open-source dataset from Kaggle. News articles in this dataset are collected from 244 websites scraped by the BS Detector tagged as “Bullshit” from past 30 days it ran. BS Detector is a chrome extension by Daniel Sieradski. This extension helped us in pulling data using an API. Data is coming from their web crawler which is a bot that browses internet for the purpose of web indexing. 
This data in this dataset contains around 18000 fake and 16000 true news articles. It also incudes 15 attributes for fake news like title, content, publishers, publish date, spam score, domain ranking, language etc.
      
Below are the attributes which we are using from the dataset:
Title: This is text for the heading/title of the news article.
Content: This is the text for the details of the news article.
Publication: Name of the publishers for the news articles.
Publish date: Date for the article published.
Spam score: This is the score available for an article whether your article is a spam or not. 0.01- 0.30 is considered as low, 0.30 – 0.60 as medium and 0.60 – 1 as high spam score. We are taking only articles having low spam score.
Domain ranking: It is search engine ranking score which predicts a websites capability to rank on search engine. Range of domain ranking goes from 1 to 100.
Language: This shows the language used for the article. 

WORKFLOW:

Setup:
We used three major services from AWS: 
•	Elastic Map Reduce: It is a service which help us in creating cluster of Amazon Elastic Compute Cloud (Amazon EC2) machines which run the hosted Spark Framework. It performs the work we submit to our cluster. 
•	Sagemaker: It is a service that provides Jupyter notebooks which we can use to build, train, and deploy machine learning models.  
•	Simple Storage Service(S3): It is storage service provided by AWS which follows object storage. 
To integrate all these services, we must follow some steps.
EMR - Sagemaker:
We create an EMR cluster by setting up below configurations:
•	Software and Steps: Here, we choose the applications to run on the cluster of Amazon EC2 machines. 
•	Hardware: Here we select specifications for our master and slave machines.
•	General Cluster Settings: We attach a key pair to our cluster, for a quick access. Also, we attach S3 bucket path for log files.
•	Security: Here we EC2 key pair, security configurations and EC2 security groups. 
Sagemaker – S3:
By knowing S3 bucket path we can access our saved files from AWS S3. We use below code to load our files into our notebook.
spark.read.load(s3_bucket_path) 



MonogDB - Sagemaker:
We use MongoDB Atlas as our database service provider. 
Python:
For this we installed pymongo connector and used below mentioned code to access data in MongoDB Atlas

# Connect to MongoDB
Client=MongoClient("mongodb+srv://****:************/database.collection?retryWrites=true&w=majority")

db = client['database']
collection = db['collection']
# Insert collection
collection.insert_many(dataframe)

Pyspark:
We install mongodb and pyspark connector into our master machine so that it can communicate with each other. We set uniform resource identifier (URI) paths as well. We use below code for connecting to MongoDB through Jupyter notebook.

We are providing URI paths while building spark session:
spark = SparkSession \
.builder \
.config("spark.mongodb.input.uri",      " mongodb+srv://****:************/database.collection?retryWrites=true&w=majority ") \
.config("mongodb+srv://****:************/database.collection?retryWrites=true&w=majority ") \
    	.getOrCreate()  
#this will load data from mongoDB database 
df = spark.read.format("mongo").load()

 
This diagram explains the flow of data from start to end.
STUDYING THE DATA:
We study our dataset by visualizing its structure and key features. Below are some of the visualizations.
Describing our dataset:
 
We see the statistics of all the columns. 






Checking the distribution of spam score:
 
The above graph shows the distribution of spam score is left skewed over all records. Low spam score suggests less chances of an article of being spam.
We made word cloud on the content column which shows the most used word in our dataset
 
Data Cleaning:

We clean our dataset by doing following steps 
•	removing null values from the dataset
•	only selecting articles of English language
•	selecting articles having spam score less than 0.40
 
After this, we push our dataset into MongoDb Atlas database.

Preprocessing:

We load dataset from MongoDB Atlas to pyspark notebook. These notebooks are backed up by Apache Spark framework hosted on EMR cluster. For preprocessing we followed some procedures mentioned in below diagram:   
 
Removing Punctuation:  
We created a function that Removed punctuation using regular expression.
#function for removing punctuation 
 DefremovePunctuation(column):
return split(trim(lower(regexp_replace(concat_ws("SEPARATORSTRING", column),'[^\sa-zA-Z0-9]', ''))), "SEPARATORSTRING").alias('stopped') 

Splitting text: 
We have split title and content column based on spaces. This gave us list of all words.
#splitting sentences to words 
split_col = pyspark.sql.functions.split(df['title_p'], ' ')

Stemming: 
This function reduces the words to their root form. We use Snowball stemmer function for our preprocessing 
# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType())) 

Removing stop words: 
We remove stopwords from title and content column which gave us only relevant words to focus on.
#Remove stop words
title_sw_remover= StopWordsRemover(inputCol= 'title_stemmed', outputCol= 'title_sw_removed')

Count Vectorizer: 
We use countvectorizer function to convert a collection of text to a vector of term counts. 
#computing frequency of the words for title
title_count_vectorizer= CountVectorizer(inputCol= 'title_sw_removed', outputCol= 'tf_title')   

IDF: 
This is statistical measure which determine how significant a word is in collection of documents. 
#Computing frequency-inverse document frequency from title
title_tfidf= IDF(inputCol= 'tf_title', outputCol= 'tf_idf_title') 

Vector Assembler: 
This function combines all the features into a single feature for our model to train.
#VectorAssembler
vec_assembler= VectorAssembler(inputCols=['tf_idf_title', 'tf_idf_text'], outputCol= 'features')  

The above preprocessing techniques convert all text data into numeric form making it ready for machine learning model.

Machine learning model (logistic regression): 

One of the best machine learning models for binary classification is logistic regression. It has simple algorithm and easy to implement. It estimates the probability of relationship between the class/label attribute and other independent attributes. Due to two class labels involved in fake news detection and few independent variables involved in predicting these class labels, logistic regression model becomes perfect choice for our project.
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.3, elasticNetParam=0)  

Pipelining: 
The pipeline API of pyspark helps specifying sequence of stages where each stage can be a transformer or an estimator. Below is the code of how we set sequences for our pipeline:  
from pyspark.ml import Pipeline
lr_pipe= Pipeline(stages=[
                title_sw_remover,   #Stop words remover for title
                title_count_vectorizer,  #count vectorizer for title 
                title_tfidf,                    #IDF for title 
                text_sw_remover,    #Stop words remover for content
                text_count_vectorizer,  #count vectorizer for content 
                text_tfidf,                #IDF for content
                vec_assembler,         #Vector assembler 
                lr])                               #logistic regression model

Training and Testing data: 

We split our data into training and testing data in the ratio of 0.7:0.3. 70% of the data is used for training and 30% of the data is used for testing.
train, test= df.randomSplit([0.7, 0.3])

Cross-validation:
 It is a model validation technique to predict new data that was not used in estimating it. This generally overcomes the problem of overfitting or selection bias. For doing cross validation we set hypermeters by using ParamGridBuilder function. Code is mention for both crossvalidation with tuning hyperparameters
 #setting up hyper parameters for cross validation 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()
#Crossvalidation
from pyspark.ml.evaluation import BinaryClassificationEvaluator
crossval = CrossValidator(estimator=lr_pipe,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=4)
Training the model: 

We train our model using training data. Below is the code:
#fitting the training data into our model
lr_model=crossval.fit(train).bestModel
Testing the model: 
We tested our model by testing it with test data created earlier. Below is the code used:
predictions = lr_model.transform(test) 
Classification evaluation:
We calculate accuracy, r-square and root mean square error of the model.
Accuracy:  Accuracy for our model comes out to be 99%.  
 
r-square: The value of R-square comes out to be 93 % which signifies that 93% of the variance in our data can be explained by our model.
 
Root mean square error (rmse): The value of RMSE always lie between 0 and 1. If the value comes closer to 0, then it means the data points close to the best line of fit and if the value is closer to 1 then it means that the residuals are far away from regression line. The value for our model comes out to be 0.128 which is close to 0.
 
Testing real time data: 
We collect some of the real time news article using web scraping. We test our on these news articles. Our model detected 4 out of 5 given fake news.
   
Saving our trained model to S3 buckets: 
We save our pipelining model to S3 bucket so that we can be used directly by our web application.
lr_model.write().overwrite().save(S3_bucket_path)
FLASK: WEB APPLICATION FRAMEWORK (WSGI BASED)
We build a web application using Flask framework in pyspark. It uses WSGI (Web Server Gateway Interface) as an interface between web servers and web apps. The spark-Submit job loads our saved model from S3 bucket and use it identify the news as fake or real. The web application takes news title and content as input from the user and outputs the result showing whether its fake or real.  Below is the pyspark code for our web application.

Setup:

Install below packages on EMR EC2 master machine to start using Flask.
Python -m pip install flask
Python -m pip install flask_restful
Python -m pip install requests
Yum install httpd mod_wsgi
#Importing necessary libraries
import flask
import pyspark
import requests
from pyspark.sql.functions import lower, col
from flask import Flask, request, jsonify, render_template
from flask_restful import reqparse, abort, Api, Resource
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml import PipelineModel
import numpy as np
from pyspark.sql.types import *
#Function that makes the data inserted by user ready for the model
def test_data(test_df):
    # lowercase the title and content column
    test_df = test_df.select("*", lower(col('content'))).drop('content')
    test_df = test_df.select("*", lower(col('title'))).drop('title')
    test_df=test_df.select('*',col("lower(content)").alias("content"),
    col("lower(title)").alias("title")).drop('lower(content)', 'lower(title)')
    #spliting the title and content column
    split_col = pyspark.sql.functions.split(test_df['title'], ' ')
    test_df = test_df.withColumn("title_split",split_col).drop('title')
    split_col = pyspark.sql.functions.split(test_df['content'], ' ')
    test_df = test_df.withColumn("content_split",split_col).drop('content')
    # Stem title and content column
    stemmer = SnowballStemmer(language='english')
    stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
    test_df = test_df.withColumn("title_stemmed", stemmer_udf("title_split"))
    test_df = test_df.withColumn("content_stemmed", stemmer_udf("content_split"))

    return test_df

app = Flask(__name__)
api = Api(app)
#Loading our html page as template
@app.route('/')
def home():
    return render_template('news_prediction.html')
#Function that takes input from the web application, pass it to the model, and posts the result.
@app.route('/predict',methods=['POST'])
def predict():
    str_features=[]
    for x in request.form.values():
        str_features.append(x)
    spark = SparkSession.builder.appName("newsPrediction").getOrCreate()
    str_columns =['title','content']
    str_values=[(str_features[0],str_features[1])]
    df = spark.createDataFrame(data=str_values,schema=str_columns)
    result = test_data(df)
    lr = PipelineModel.load(S3_bucket_path)
    prediction = lr.transform(result)
    result_final=str(prediction.select("prediction").collect()[0])
    final_pred=result_final[15:16]
    if(final_pred=='1'):
    	final_pred="Fake"
    else:
    	final_pred="Real"
    return render_template('news_prediction.html', prediction_text='News is {}'.format(final_pred))
#Runs the application on local host of EMR ec2 machine
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)’
RESULTS:
Below is the web application build with flask that shows the input taken by the user and result of our model.
 

 
PROBLEMS FACED AND THEIR SOLUTIONS:
Problem 1:
The communication between Amazon EMR cluster’s EC2 master machine and the MongoDB was a very difficult task. The approach we used to connect to MongoDB and load the data into the spark dataframe involved a lot of steps that have not been done before. We are the pioneers of the way we used our MongoDB database to speak to our cluster’s master machine.
Solution 1:
The steps were:
Permitting the root login on the cluster by using below commands
•	hadoop:172.31.28.6 ~] sudo nano /etc/ssh/sshd_config
•	#Authentication
•	PermitRootLogin yes
•	ctrl X
•	hadoop:172.31.28.6 ~]sudo nano /root/.ssh/authorized_keys
•	comment first line and press enter before ssh-rsa
Login as root user
Go to the below path
•	root:172.31.28.6 ~]cd /usr/lib/spark/jars/
Install below jar files
•	wget https://repo1.maven.org/maven2/org/mongodb/spark/mongo-spark-connector_2.11/2.4.2/mongo-spark-connector_2.11-2.4.2.jar
•	wget https://repo1.maven.org/maven2/org/mongodb/mongo-java-driver/3.8.0/mongo-java-driver-3.8.0.jar
Problem 2:
Using our html page as template to build our application through Flask was a tricky part for us. We faced issues while installing the right web application servers and interface. Loading our model to the script and using it with the values inserted on the web page took more time than we imagined. 
Solution 2:
Below are the commands to install web server and interface on Amazon EMR cluster’s master Ec2 machine that worked for us:
Yum install httpd mod_wsgi
Making a separate directory for templates to load html file.
Mkdir templates 
Place the html code for the web application inside this folder.
CONCLUSION:
Building an end-to-end pipeline machine learning model using Apache spark framework and creating a web application that uses the model to identify fake news, was a great learning experience for us. Specially achieving everything on AWS, enhanced our skills in cloud computing. We are able to achieve good accuracy for our model, but our model is predictive of US region specific news. We can improve our model by training it on news articles which belong to other countries. It would require gathering lot of news articles over the world. 
The services provided by AWS were efficient, but it can cost a lot if not used wisely. Having a proper knowledge about the costs of services would be very beneficial before using cloud services.

REFERENCES:
https://spark.apache.org/docs/2.2.0/ml-pipeline.html
https://www.kaggle.com/mrisdal/fake-news
https://spark.apache.org/docs/latest/api/python/pyspark.ml.html
https://en.wikipedia.org/wiki/Apache_Spark
https://spark.apache.org/docs/latest/
https://www.rogerebert.com/reviews/the-great-hack-2019
https://www.forbes.com/sites/bernardmarr/2015/09/30/big-data-20-mind-boggling-facts-everyone-must-read/?sh=4f7cf80217b1
https://www.red-gate.com/simple-talk/blogs/harsh-reality-behind-big-data-misuse/


 
