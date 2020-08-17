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

class tweet_listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]

        print(tweet)

        return True

    def on_error(self, status):
        print(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--movie',
                action="store", dest="movie",
                help="pick a movie")

    args = parser.parse_args()

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

    if args.movie:
        twitterStream = Stream(auth, tweet_listener())
        twitterStream.filter(track=[args.movie])
    else:
        print("You need to request a movie to rate.")


# print(s.sentiment("I’m like an hour fifteen into ONCE UPON A TIME... IN HOLLYWOOD and it’s beautiful and charming and absolutely nothing has happened yet"))

# print(s.sentiment("The movie is what, 3 hours? And the first 2 hours 45 min was a bit ramble-y but promised either an exciting convergence of the A and B plots, or an emotional reckoning of an old star being given a second chance. We got neither and in the last 15 minutes we got excessive gore as two grown men fight and kill teenagers. Which. I should have seen coming I GUESS."))