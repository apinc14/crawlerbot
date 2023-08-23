import praw
import pymysql
import os 
from datetime import datetime, timedelta
import torch
from transformers import BertTokenizer,  RobertaForSequenceClassification, RobertaTokenizer, BertForTokenClassification, pipeline, AutoTokenizer
from torch import cuda
import pandas as pd
from transformers import pipeline
import nltk
from rake_nltk import Rake
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.disable(logging.CRITICAL)
from PIL import Image
from io import BytesIO
import requests 
from urllib.parse import urlparse
import time 
import json
from langdetect import detect


reddit = praw.Reddit(
    client_id='r7LUrKrxa1Sk8cn9CqULbw',
    client_secret='npm2YPEat8__5vELdtdW4zzfghLCEw',
    user_agent='windows:Crawler:v1.00 by /u/Nervous-profile',
    )


def saveToDB(allPosts):
    host = 'localhost'
    port = 3306
    user = 'root'
    password = 'snook1sm00sh0Osmoozh'
    db_name = 'weblyticsDB'

    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        db=db_name,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    for x in allPosts:
        print(x)

       
    print("attempted save")
    
    try:
        formatted_allPosts = []  # Define the list outside the try block
        with connection.cursor() as cursor:
            insert_query = """
                INSERT INTO posts
                (id, post_title, post_content, post_score, num_comments, creation_time, 
                subreddit_name, sentiment, entitiesArray, keywordsArray, upVRatio, 
                ups, media)
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            print("insert 1 --- done")
            insert_totals_query = """
                INSERT INTO totals
                (entity, score)
                VALUES
                (%s, %s)
            """

            for post in allPosts:
                formatted_post = (
                    post[0],
                    post[1],
                    None if post[2] == "" else post[2],  # Handle empty content
                    post[3],
                    post[4],
                    post[5],
                    post[6],
                    json.dumps(post[7]),
                    json.dumps(post[8]),  # Serialize entitiesArray as string
                    json.dumps(post[9]),  # Serialize keywordsArray as string
                    post[10],
                    post[11],
                    None if post[12] == "" else post[12],  # Handle empty media
                )
                formatted_allPosts.append(formatted_post)
                print("before for")

            for formatted_post in formatted_allPosts:
                try:
                    
                    if post[8] != None:
                        print("8 IS", post[8])
                        print(type(post[7]))
                        
                        all = []
                        for x in post[8]:
                            totals_data = (x, 1)
                            all.append(totals_data)
                            print("sentiment type",type(x), "--", x)
                              # To debug and check the type of x
                        cursor.executemany(insert_totals_query, all)
                        

                    print("try start")
                    print(formatted_post[0], formatted_post[1])
                    print("-")
                    #cursor.execute(insert_query, formatted_post)  # Execute the first SQL query
                    print("-")
                    connection.commit()
                    print("Data inserted successfully!")
                except pymysql.err.IntegrityError as e:
                    if e.args[0] == 1062:  # Duplicate entry error
                        print("Duplicate entry. Skipping...")
                    else:
                        print("Error:", e)
    except Exception as e:
        print("Error:", e)
    finally:
        cursor.close()

        


   


  





# Set up PRAW with your Reddit API credentials




def getSentiment(post_content):
    print("-----")
    print(type(post_content))
    print(type(post_content))
    print(type(post_content))
    print(type(post_content))
    print(type(post_content))
    # Load the sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # Perform sentiment analysis on the input text
    results = sentiment_analyzer(post_content)
    for x in results:
        print("sentiment--", x)

    return results






def get_entities_batch(input_sentences):
    
    tokenizer = BertTokenizer.from_pretrained("dslim/bert-large-NER")
    model = BertForTokenClassification.from_pretrained("dslim/bert-large-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    print("------------------")
    ner_results_list = nlp(input_sentences)
    formatted_entities_batch = []

    for ner_results in ner_results_list:
        formatted_entities = []
        formatted_categories = []
        current_entity = ''
        current_word = ''

        for result in ner_results:
            entity_type = result['entity'][2:]  # Get the 3-letter entity category
            word = result['word'].replace('##', '')
            
            if result['entity'].startswith('B-'):
                if current_entity and current_word:
                    formatted_categories.append(current_entity)
                    formatted_entities.append(current_word)
                current_entity = entity_type
                current_word = word
            elif result['entity'].startswith('I-'):
                current_word += word
            else:
                continue
        
        if current_entity and current_word:
            formatted_categories.append(current_entity)
            formatted_entities.append(current_word)
        
        combined_entities = []
        combined_categories = []
        i = 0
        while i < len(formatted_entities):
            entity = formatted_entities[i]
            category = formatted_categories[i]
            
            if i < len(formatted_entities) - 1 and formatted_categories[i] == formatted_categories[i + 1]:
                i += 1
                while i < len(formatted_entities) and formatted_categories[i] == 'I-' + category:
                    entity += formatted_entities[i]
                    i += 1
            else:
                i += 1
            
            combined_entities.append(entity)
            combined_categories.append(category)
        
        formatted_entities_batch.append((combined_entities, combined_categories))
    print("------------------")
    return formatted_entities_batch

def get_entities(input_sentences):
    # Split input sentences into batches
    batch_size = 8  # You can adjust this based on memory and performance
    input_batches = [input_sentences[i:i+batch_size] for i in range(0, len(input_sentences), batch_size)]
    
    formatted_entities_batch = get_entities_batch(input_batches)
    
    # Flatten the results and return
    formatted_entities = []
    formatted_categories = []
    print("------------------")
    for entities, categories in formatted_entities_batch:
        formatted_entities.extend(entities)
        formatted_categories.extend(categories)
    
    return formatted_entities, formatted_categories


def rake(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    for x in keywords:
        print(x)
    return keywords

def crawl_get_values(subreddit_name):
        num_posts = 1
        subreddit = reddit.subreddit(subreddit_name)

        # Get new posts from the subreddit
        num_posts = 10  # Adjust the number of posts you want to retrieve
        for submission in subreddit.new(limit=num_posts):
            submission_attributes = vars(submission)
            for attribute, value in submission_attributes.items():
                print(f"{attribute}: {value}")
            print("=" * 50)


def count_posts_from_yesterday(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    yesterday = datetime.now() - timedelta(days=1)
    start_of_yesterday = int(yesterday.timestamp())
    
    count = 0
    for submission in subreddit.new(limit=None):
        if submission.created_utc < start_of_yesterday:
            break
        count += 1
    
    return count


def crawl_reddit_subreddit(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    allPosts = []
    yesterday = datetime.now() - timedelta(days=1)
    start_of_yesterday = int(yesterday.timestamp())
    num_posts = count_posts_from_yesterday(subreddit_name)
    print(f"Number of posts from yesterday in /r/{subreddit_name}: {num_posts}")
    print(num_posts, " posts")
    print("crawling")
    count = 1
    num_posts = 2
    try:
         
        # Now you can insert data into the 'newsarticle' table
         for submission in subreddit.new(limit=num_posts):
            if submission.created_utc < start_of_yesterday:
                 break
            media=None 
            if submission.is_created_from_ads_ui:       
                return
            if hasattr(submission, 'crosspost_parent'):
                return
            if submission.media is not None:
             # Check if media is a valid URL
                media_info = submission.media
                for x in media_info:
                    print(x)
                if 'fallback_url' in media_info:
                    media = media_info['fallback_url']
                    
                    # Now you can use parsed_url for further processing
                else:
                    print("No media URL found")
            language = detect(submission.title)
            print(language)
            if language != "en":
                return    
            
                    
            r = Rake()   
            sentiment = ""
            entitiesArray = []
            
            post_title = submission.title
            post_content = submission.selftext
            post_score = submission.score
            num_comments = submission.num_comments
            id = submission.id
            subreddit_name = submission.subreddit.display_name
            upVRatio  = submission.upvote_ratio
            ups = submission.ups
            unix_timestamp = submission.created
            creation_time = datetime.fromtimestamp(unix_timestamp)
            formatted_creation_time = creation_time.strftime('%Y-%m-%d %H:%M:%S')
            
            for x in id:
                print("---",x)
            print(creation_time)
            id = id[0]+id[1]+id[2]+id[3]+id[4]+id[5]+id[6]
            
            if post_content:
                print(type(post_content))
                print(type(post_content))
                print(type(post_content))
                print(type(post_content))
                print(type(post_content))
                print("content is ", post_content)
                sentiment = getSentiment(post_content)
                entitiesArray = get_entities(post_content)
                r.extract_keywords_from_text(post_content)
            else:
                print("title - ", post_title)
                sentiment = getSentiment(post_title)
                print("entities")
                entitiesArray = get_entities(post_title)
                r.extract_keywords_from_text(post_title)
            keywordsArray = r.get_ranked_phrases()
            posts_data = [id, post_title, post_content, post_score, num_comments,
            formatted_creation_time,  subreddit_name, sentiment, entitiesArray, keywordsArray, upVRatio, ups, media  ]
            allPosts.append(posts_data)
            count +=1
            print(count)

    finally: 
        saveToDB(allPosts)
        print("ran")
            
def timeLoop():
    subreddit_names = [ "Economy", "Politics", "Business","News", "Technology","Gadgets", "financialindependence", "Environment", "popculturechat", "Entertainment" ]

    
    requests_per_minute = 7
    # Number of requests to make every minute
    minutes_to_run = 1440

    # Number of hours to run the process
    


    # Loop for the specified number of hours
   
    # Loop to make requests for each minute within the hour
    for _ in range(minutes_to_run):
        # Loop to make requests for each minute
        for _ in range(requests_per_minute):
            
            
            subreddit_name = subreddit_names[_ % len(subreddit_names)]
            print("requested ", subreddit_name)
            print("requested ", subreddit_name)
            print("requested ", subreddit_name)        
            crawl_reddit_subreddit(subreddit_name)
           
        
        # Sleep for 1 minute before the next iteration
            

    if _ < hours_to_run - 1:
        ("nap time Zzz")
        time.sleep(3600)  # Sleep for 1 hour

  
if __name__ == "__main__":
    #timeLoop()
    crawl_reddit_subreddit("Economy")
    #crawl_get_values("News")

