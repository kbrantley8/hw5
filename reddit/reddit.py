import requests
import praw
import re

username = 'butternife17'
password = open("password.txt", "r").read().strip('\n')
app_id = 'VE2_SDrfxa6Auw'
app_secret = 'GE4LhAymCmgXVLAjVLP_OaCa2Ko'
app_name = 'Reddit Poster'

subreddit_name = 'wallstreetbets'
filepath = "../models/data/kory_data.csv"

def writeLatestPosts(firsttime=False):
    f = open(filepath, "a", encoding="utf-8")
    if (firsttime): 
        f.write("text,label\n")

    reddit = praw.Reddit(
        client_id=app_id,
        client_secret=app_secret,
        user_agent=username,
    )

    subreddit = reddit.subreddit(subreddit_name)

    for submission in subreddit.hot(limit=1):
        submission = reddit.submission(id=submission.id)
        submission.comment_sort = "new"
        submission.comments.replace_more(limit=0)
        for c in submission.comments:
            s = re.sub('([.,!?()])', r' \1 ', c.body)
            s = re.sub('\s{2,}', ' ', s)
            f.write('"' + s + '",0\n')

    f.close()
    # removeDuplicates()

def removeDuplicates():
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines() 


    lines = list(set(lines))
    with open(filepath, 'w', encoding="utf-8") as f:
        f.write("text,label\n")
        for item in lines:
            f.write("%s\n" % item)
    
writeLatestPosts()

def findTickers():
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines() 
        
# seen = set()
# result = []
# for item in lines:
#     if item not in seen:
#         seen.add(item)
#         result.append(item)
# print(result)

