import praw
import pymysql
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForTokenClassification, pipeline
from rake_nltk import Rake
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.disable(logging.CRITICAL)
import time 
import json
from langdetect import detect



reddit = praw.Reddit(
    client_id='r7LUrKrxa1Sk8cn9CqULbw',
    client_secret='npm2YPEat8__5vELdtdW4zzfghLCEw',
    user_agent='windows:Crawler:v1.00 by /u/Nervous-profile',
)

def summarize(post_content):
    

    word_count = len(post_content.split())
    if word_count>99:

        max_length_162_percent = int(word_count * 0.939)
        max_length_37_percent = int(word_count * 0.639)
        return None

    else:
        return post_content
   
    
   

def saveToDB(allPosts):
    print("very start save")
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
                INSERT INTO totals (entity, score, type)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE score = score + 1;
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
                    json.dumps(post[7]) if post[7] is not None else None,
                    json.dumps(post[8]) if post[8] is not None else None,
                    json.dumps(post[9]) if post[9] is not None else None,
                    post[10],
                    post[11],
                    None if post[12] == "" else post[12],  # Handle empty media
                )

                formatted_allPosts.append(formatted_post)
                

            for formatted_post in formatted_allPosts:
                print("big loop")
                try:
                    print("8 IS", post[8])
                    if post[8] is not None and len(post[8]) != 0:

                        print("8 IS", post[8][0])
                        print("8 IS", post[8][1])
                        
                        
                        all = []
                        count = 0
                        for x in post[8]:
                            print("loop")
                            ent = post[8][0][count]
                            type1 =  post[8][1][count]
                            totals_data = [ent, 1, type1]
                            all.append(totals_data)
                            print(totals_data, "data is")
                            print("change here")
                            count += 1
                        
                        cursor.executemany(insert_totals_query, all)
                        print("end of if")
                    
                        

                    print("try start")
                    print(formatted_post[0], formatted_post[1])
                    
                    cursor.execute(insert_query, formatted_post)  # Execute the first SQL query
                    
                    connection.commit()
                    
                except pymysql.err.IntegrityError as e:
                    if e.args[0] == 1062:  # Duplicate entry error
                        print("Duplicate entry. Skipping...")
                    else:
                        print("Error:", e)
               
    except Exception as e:
        print("Error:", e)
    finally:
        cursor.close()

        


def get_entities_batch(input_sentences):
    
    tokenizer = BertTokenizer.from_pretrained("dslim/bert-large-NER")
    model = BertForTokenClassification.from_pretrained("dslim/bert-large-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    
    ner_results_list = nlp(input_sentences)
    formatted_entities_batch = []

    for ner_results in ner_results_list:
        formatted_entities = []
        
        formatted_categories = []
        current_entity = ''
        current_word = ''
        
        for result in ner_results:
            # Handle both cases: 'entity_group' and 'entity'
            if 'entity_group' in result:
                entity_key = 'entity_group'
            else:
                entity_key = 'entity'
                
            entity_type = result[entity_key][2:]  # Get the 3-letter entity category
            word = result['word'].replace('##', '')

            if result[entity_key].startswith('B-'):
                if current_entity and current_word:
                    formatted_categories.append(current_entity)
                    formatted_entities.append(current_word)
                current_entity = entity_type
                current_word = word
            elif result[entity_key].startswith('I-'):
                current_word += ' ' + word  # Add a space before appending the next word
            else:
                continue
            
        
        if current_entity and current_word:
            formatted_categories.append(current_entity)
            formatted_entities.append(current_word)
        
        combined_entities = []
        combined_categories = []
        for entity, category in zip(formatted_entities, formatted_categories):
            if entity and category:
                combined_entities.append(entity)
                combined_categories.append(category)
            
        formatted_entities_batch.append((combined_entities, combined_categories))
    
    return formatted_entities_batch


def get_entities(input_sentences):
    print(input_sentences)
    print(len(input_sentences))
    print(len(input_sentences))
    batch_size = 8
    input_batches = [input_sentences[i:i+batch_size] for i in range(0, len(input_sentences), batch_size)]
    
    formatted_entities_batch = get_entities_batch(input_batches)
    
    formatted_entities = []
    formatted_categories = []
    
    for entities, categories in formatted_entities_batch:
        formatted_entities.extend(entities)
        formatted_categories.extend(categories)
        
    return formatted_entities, formatted_categories






# Set up PRAW with your Reddit API credentials





def getSentiment(post_content):
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer=model_path, max_length=512, truncation=True)
    
    # Perform sentiment analysis on the input text
    results = pipe(post_content)  # Use 'text' instead of 'inputs'
    
    for x in results:
        print("sentiment--", x)
    
    return results




def rake(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    
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


def count_posts_from_three_days_ago(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    
    # Get the current time
    now = datetime.now()

    # Calculate the timestamp for the start of 3 days ago
    three_days_ago = now - timedelta(days=1)
    start_of_three_days_ago = int(three_days_ago.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    
    count = 0
    for submission in subreddit.new(limit=None):
        if submission.created_utc < start_of_three_days_ago:
            break
        count += 1
    
    return count



def crawl_reddit_subreddit(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    allPosts = []
    yesterday = datetime.now() - timedelta(days=1)
    start_of_yesterday = int(yesterday.timestamp())
    num_posts = count_posts_from_three_days_ago(subreddit_name)
    print(f"Number of posts from yesterday in /r/{subreddit_name}: {num_posts}")
    print(num_posts, " posts")
    
    count = 1
    num_posts = 5
    
        
    # Now you can insert data into the 'newsarticle' table
    for submission in subreddit.new(limit=num_posts):
        try:
            if submission.created_utc < start_of_yesterday:
                break
            media=None 
            if submission.is_created_from_ads_ui:       
                continue
            if hasattr(submission, 'crosspost_parent'):
                continue
            if submission.media is not None:
            # Check if media is a valid URL
                media_info = submission.media
                for x in media_info:
                    print(x)
                if 'fallback_url' in media_info:
                    media = media_info['fallback_url']
                    
                    # Now you can use parsed_url for further processing
                
            language = detect(submission.title)
            
            if language != "en":
                continue    
            
                    
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
            
           
                
            print(creation_time)
            id = id[0]+id[1]+id[2]+id[3]+id[4]+id[5]+id[6]
            

            
            if post_content:
                
                
                   
                
                sentiment = getSentiment(post_content)
                print("content")
                entitiesArray = get_entities(post_content)
                r.extract_keywords_from_text(post_content)
                post_content = summarize(post_content)
                
                print(type(post_content))
                
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
            print("done")
    print(len(allPosts))
    saveToDB(allPosts)        
   
            
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
            

    if _ < minutes_to_run - 2:
        ("nap time Zzz")
        time.sleep(3600)  # Sleep for 1 hour
        minutes_to_run = 1440

  
if __name__ == "__main__":
    #timeLoop()
    crawl_reddit_subreddit("Economy")
    #crawl_get_values("News")
    

