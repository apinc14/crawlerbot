import praw
import pymysql
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForTokenClassification, pipeline , BartTokenizer, BartForConditionalGeneration
import psutil
from rake_nltk import Rake
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.disable(logging.CRITICAL)
import time 
import json
from langdetect import detect
import re
import asyncio
import aiohttp
import math
# Function to get CPU and memory usage
def get_system_stats():
    # CPU usage as a percentage
    cpu_usage = psutil.cpu_percent(interval=1)  # You can adjust the interval as needed

    # Memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    return cpu_usage, memory_usage


# Function you want to call (replace this with your function)



# Token bucket rate limiting class
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill_time = time.time()

    async def consume(self):
        while True:
            current_time = time.time()
            time_elapsed = current_time - self.last_refill_time

            # Refill tokens if needed
            if time_elapsed > 1.0:
                tokens_to_add = int(time_elapsed * self.rate)
                self.tokens = min(self.capacity, self.tokens + tokens_to_add)
                self.last_refill_time = current_time

            if self.tokens >= 1:
                self.tokens -= 1
                return True  # Token available, allow the call
            else:
                await asyncio.sleep(0.1)  # Sleep briefly and check again

import asyncio

async def run_task(subreddit_names, sem):
    rate_limit = TokenBucket(rate=60, capacity=60)  # Limit to 60 calls per minute

    async def fetch_data(name):
        async with sem:# Simulate some work
            await crawl_reddit_subreddit(name)

    while True:
        tasks = []
        for name in subreddit_names:
            if await rate_limit.consume():
                print("loooop k329")
                print(name)
                print(name)
                print(name)
                print(name)
                print(name)
                print(name)
                print(name)
                task = asyncio.create_task(fetch_data(name))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks)  # Run tasks concurrently
            
# Function to run the asynchronous task
async def run_task(subreddit_names, sim):
    rate_limit = TokenBucket(rate=60, capacity=60)  # Limit to 60 calls per minute

    while True:
        tasks = []
        for name in subreddit_names:
            if await rate_limit.consume():
              
                async with sim:
                    tasks.append(crawl_reddit_subreddit(name))  # Create tasks for each subreddit

        if tasks:
            await asyncio.gather(*tasks)  # Run tasks concurrently

# Function to get CPU and memory usage
def get_system_stats():
    # CPU usage as a percentage
    cpu_usage = psutil.cpu_percent(interval=1)  # You can adjust the interval as needed

    # Memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    return cpu_usage, memory_usage



def summarize(post_content):
    word_count = len(post_content.split())
    # Load pre-trained BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    

    # Tokenize and generate summary
    input_ids = tokenizer.encode(post_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

    # Decode the generated summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("Original text:")
    print(post_content)
    print("\nSummarized text:")
    if word_count > 199:
        max_length_162_percent = int(word_count * 0.939)
        max_length_37_percent = int(word_count * 0.639)
        print("summarize ",summary_text,"!!!!!!!!!!!!!!!!!!")
        return summary_text  
    else:
        return post_content
    
   

def saveToDB(allPosts):
    print(allPosts)
    print(len(allPosts))
    try:
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
            insert_totals_query = """
                INSERT INTO totals (entity, score, type)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE score = score + 1;
            """

            for post in allPosts:
                try:
                    formatted_post = (
                        post[0],
                        post[1],
                        None if post[2] == "" else post[2],
                        post[3],
                        post[4],
                        post[5],
                        post[6],
                        json.dumps(post[7]) if post[7] is not None else None,
                        json.dumps(post[8]) if post[8] is not None else None,
                        json.dumps(post[9]) if post[9] is not None else None,
                        post[10],
                        post[11],
                        None if post[12] == "" else post[12],
                    )

                    formatted_allPosts.append(formatted_post)

                    entities_json = formatted_post[8]  # Get the JSON string
                    entities = json.loads(entities_json)  # Parse it into a Python data structure

                    if entities is not None and len(entities) > 1 and len(entities[0]) == len(entities[1]):
                        for i in range(len(entities[1])):
                            ent = entities[0][i]  # Access elements from the parsed data
                            type1 = entities[1][i]  # Access elements from the parsed data
                            totals_data = (ent, 1, type1)
                            print(totals_data)
                            cursor.execute(insert_totals_query, totals_data)
                            print("Executed insert_totals_query")
                    print(formatted_post)
                    cursor.execute(insert_query, formatted_post)
                    connection.commit()
                    print("Committed to the database")

                except pymysql.err.IntegrityError as e:
                    if e.args[0] == 1062:
                        print("Duplicate entry. Skipping...")
                    else:
                        print("Error:", e)
                except Exception as e:
                    print("Error in inner loop:", e)

    except Exception as e:
        print("Error in outer loop:", e)
    finally:
        cursor.close()

    






def find_and_split_sentences(text):
    sentence_endings = re.finditer(r'[.!?;|)]\s+', text)

    for x in sentence_endings:
        print(x,"endigns")
    sentence_endings_indexes = [match.start() for match in sentence_endings]
    print(sentence_endings_indexes)
    if sentence_endings_indexes == []:
        sentence_endings_indexes.append(len(text) - 1)
    
    # Check for closely spaced punctuation marks
    closely_spaced_indexes = []
    for i in range(1, len(sentence_endings_indexes)):
        if sentence_endings_indexes[i] - sentence_endings_indexes[i - 1] <= 2:
            closely_spaced_indexes.append(sentence_endings_indexes[i - 1])
            closely_spaced_indexes.append(sentence_endings_indexes[i])
    
    # Combine closely spaced punctuation marks into sentences
    sentences = []
    char_index = 0
    lengths = []
    sentence_counter = 0 

    for idx, end_index in enumerate(sentence_endings_indexes):
        sentence = text[char_index:end_index + 1].strip()
        sentence_counter += 1 
        print(sentence_counter)
        print("len sentence",len(sentence))
        print(sentence)
        if sentence.endswith("'s") and idx + 1 < len(sentence_endings_indexes):
            next_sentence_start = sentence_endings_indexes[idx + 1] + 1
            next_sentence = text[end_index + 1:next_sentence_start].strip()
            print("''''''''''''")
            sentence += " " + next_sentence
            char_index = next_sentence_start
        else:
            print(end_index)
            char_index = end_index + 1

        l = len(sentence)
        print("l = = = = = ", l)
        
        if l == 0:
            print(00000000)
            l = 1
        lengths.append(l)
        sentences.append(sentence)
        
        # Check if the current index is a part of closely spaced punctuation
        if end_index in closely_spaced_indexes:
            char_index = end_index + 1

    if char_index < len(text):
        sentences.append(text[char_index:].strip())
        sentence_counter += 1  # Increment the counter when appending the last sentence

    return sentences, lengths, sentence_counter









def get_entities_batch(input_sentences):
    print("sen its", input_sentences)
    print(len(input_sentences))
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
            word = result['word']
            addSpace = False
            if all(val in word for val in '##'):
                word = result['word'].replace('##', '')
                print("replaced")
            else:
                addSpace = True
            
            if result['entity'].startswith('B-'):
                if current_entity and current_word:
                    formatted_categories.append(current_entity)
                    formatted_entities.append(current_word)
                current_entity = entity_type
                current_word = word
            elif result['entity'].startswith('I-'):
                if addSpace:
                    current_word += ' ' + word
                else:
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
    print("-")
    sentences, lengths, sentence_counter = find_and_split_sentences(input_sentences)
    print(sentence_counter)
    print("-")
    print("lens", lengths)
    print("len sentences",len(sentences))
    print(sentences)
    
    for index, sentence in enumerate(sentences):
        
        # Your processing logic for each sentence
        # 'index' will hold the current index of the sentence
        # 'sentence' will hold the current sentence in each iteration
        # Add your code here to process 'sentence' and use 'index' if needed
        print("Index:", index,"==================")
        
        print("sentence len", len(sentence))
        
        

        
        print("lengths len", len(lengths))
        print(lengths, "lengths")
        print(lengths[index])
        batch_size = (lengths[index] )  # You can adjust this based on memory and performance
        print("b size",batch_size)
        input_batches = [input_sentences[i:i+batch_size] for i in range(0, len(input_sentences), batch_size)]
        
        formatted_entities_batch = get_entities_batch(input_batches)
          # Flatten the results and return
        formatted_entities = []
        formatted_categories = []
        print("------------------")
        for entities, categories in formatted_entities_batch:
            formatted_entities.extend(entities)
            formatted_categories.extend(categories)  
        
    
  
        
    print("ents are ", formatted_entities)
    return formatted_entities, formatted_categories







def getSentiment(post_content): 
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer=model_path, max_length=512, truncation=True)
    
    # Perform sentiment analysis on the input text
    results = pipe(post_content)  # Use 'text' instead of 'inputs'
    
    for x in results:
        print("sentiment--", x)
    
    return results



def remove_urls(text):
    print(type(text))
    # Define a pattern to match URLs
    url_pattern = re.compile(r'\b(?:https?://|www\.)\S+\b')
    
    # Replace URLs with an empty string
    text_without_urls = re.sub(url_pattern, '', text)
    print(text_without_urls, "no urls")
    return text_without_urls




def rake(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords_with_scores = r.get_ranked_phrases_with_scores()  # Updated method
    
    return keywords_with_scores


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

def define_reddit():
    reddit = praw.Reddit(
        client_id='r7LUrKrxa1Sk8cn9CqULbw',
        client_secret='npm2YPEat8__5vELdtdW4zzfghLCEw',
        user_agent='windows:Crawler:v1.00 by /u/Nervous-profile',
    )
    return reddit
def count_posts_from_three_days_ago(subreddit_name):
    print("start count_")
    reddit = define_reddit()
    subreddit = reddit.subreddit(subreddit_name)
    
    # Get the current time
     

    # Calculate the timestamp for the start of 3 days ago
    three_days_ago = datetime.now() - timedelta(days=3)
    print("next")
    start_of_three_days_ago = three_days_ago.replace(hour=0, minute=0, second=0, microsecond=0)
    print("get timestamp")
    timestamp = int(start_of_three_days_ago.timestamp())
# Convert to a Unix timestamp (if needed).timestamp()))
    print(start_of_three_days_ago)
    count = 0
    print("start for")
    for submission in subreddit.new(limit=None):
        print(submission)
        if submission.created_utc < timestamp:
            break
        count += 1

    print("returned count from time",count)
    return count,  start_of_three_days_ago



def crawl_reddit_subreddit(subreddit_name):
    print("start crawl")
    reddit = define_reddit()

    subreddit = reddit.subreddit(subreddit_name)
    allPosts = []
    
    
    num_posts, post_time = count_posts_from_three_days_ago(subreddit_name)
    print("p time", post_time)
    print(f"Number of posts from yesterday in /r/{subreddit_name}: {num_posts}")
    round = math.ceil(num_posts / 100)
    print("roundeed", round,"k328")
    current_value = 0
    count = 1
    for _ in range(round):
        print("start")
        print(current_value)
        current_value += 100
        print("-=-=-=-=-=-")  
        if current_value >= num_posts:
            break
        print("-=-=-=-=-=-")   
        remaining_value = num_posts - current_value
        if remaining_value > 0:
            current_value += remaining_value
        print(f"Reached {current_value}.")
        print("-=-=-=-=-=-")  
        postsToRun=0
        if remaining_value <= 100 :
            postsToRun = 100
           
        else:
            postsToRun =remaining_value
        print("-=-=-=-=-=-")   
        print(postsToRun)
        print("start for crawl")    
        # Now you can insert data into the 'newsarticle' table
        for submission in subreddit.new(limit=postsToRun):
            try:
                print("for try")
                print(type(post_time))
                submission_time = datetime.utcfromtimestamp(submission.created_utc)
                if submission_time < post_time:
                    print("break")
                    break
                print("if")
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
                print("language")    
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
                

                keywordsArray = []
                
                print("og content - -- - - - - - - -")
                print("ps  d", post_content)
                runT = False
                
                if post_content != None and len(post_content) > 0:
                    print(len(post_content))
                    print("pc is ",post_content)
                    post_content = summarize(post_content)
                    
                    print("ran if ")
                    print("e run")
                    print(": F")
                    print("c", post_content)
                    print("t", post_title)
                    post_content = remove_urls(post_content)
                    print(post_content)
                    print(len(post_content))
                    if len(post_content) <= 0 or post_content == None:
                        runT = True
                    if runT != True:
                        sentiment = getSentiment(post_content)
                        entitiesArray = get_entities(post_content)
                        keywordsArray = rake(post_content)  # Corrected line
                        
                        print("content - ", post_content)
                        print(type(post_content))
                else:
                    runT = True
                if runT:
                    
                    print("t run")
                    print("t", post_title)
                    post_content = remove_urls(post_title)
                    sentiment = getSentiment(post_title)
                    entitiesArray = get_entities(post_title)
                    keywordsArray = rake(post_title)
                    print("content - ", post_title)
                    print(type(post_title))
            
                print("appended ---------------______--_______----_")
                posts_data = [id, post_title, post_content, post_score, num_comments,
                formatted_creation_time,  subreddit_name, sentiment, entitiesArray, keywordsArray, upVRatio, ups, media  ]
                allPosts.append(posts_data)
                count +=1
                print(count)
            finally: 
                print("done")
        print(len(allPosts))
        saveToDB(allPosts)        
   
  
   

  
if __name__ == "__main__":
    concurrency_limit = 5  # Set the concurrency limit here
    sem = asyncio.Semaphore(concurrency_limit)
    loop = asyncio.get_event_loop()
    subreddit_names = [ "Economy", "Politics", "games", "Business","News", "Technology","damnthatsinteresting","Gadgets", "financialindependence", "Environment", "popculturechat", "Entertainment", "worldnews", "food","nottheonion", "books", "space",  "art","explainlikeimfive", "mildlyinteresting","sports" , "documentaries","upliftingnews" , "history","television" , "internetisbeautiful","wallstreetbets" , "fitness","travel",  "place","cryptocurrency",  "gardening","programming", "stocks","youshouldknow" , "outoftheloop","biology" , "teenagers","offmychest", "parenting", "bestof" , "lifestyle", "home" , "mental", "hardware" , "data", "formen"   , "forwomen", "weather"  , "news", "qoutes" , "advice", "answers" , "youshouldknow", "education" , "wikipedia", "economics"  ]
    asyncio.ensure_future(run_task(subreddit_names,sem))
    loop.run_forever()  # Start the rate-limited task
    while True:
        cpu_usage, memory_usage = get_system_stats()
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {memory_usage}%")
        
        # Add your logic here to respond to CPU and memory usage

        # Sleep for a while to avoid excessive polling
        print("sleeping ==================")
        time.sleep(10)  # Sleep for 10 seconds (adjust as needed)

    
    #crawl_get_values("News")
    

