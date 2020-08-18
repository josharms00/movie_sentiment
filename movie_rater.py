import sentiment as s
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import argparse

# IMPORTANT: must have twitter dev account and access tokens to use this
def getKeys():
    # replace with own path to keys
    f_keys = open("C:/Users/josh/Projects/tokens.txt", "r").read()

    keys = f_keys.split("\n")

    return keys[:4]

ckey, csecret, atoken, asecret = getKeys()

pos = 0
neg = 0
cnt = 0
total = 1000

class tweet_listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]

        res = s.sentiment(tweet)

        if res == 'pos':
            pos +=1
        else:
            neg += 1

        cnt += 1

        if total == cnt:
            print("Percentage of positive reviews: ", (pos/total))
            return False
        else: 
            return True

    def on_error(self, status):
        print(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--movie',
                action="store", dest="movie",
                help="pick a movie")

    parser.add_argument('-t', '--tweets',
                action="store", dest="tweets",
                help="number of tweets to look at.")

    args = parser.parse_args()
    
    total = args.tweets

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

    if args.movie:
        twitterStream = Stream(auth, tweet_listener())
        twitterStream.filter(track=[args.movie])
    else:
        print("You need to request a movie to rate.")
