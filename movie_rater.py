import sentiment as s
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

# IMPORTANT: must have twitter dev account and access tokens to use this
def getKeys():
    f_keys = open("C:/Users/josh/Projects/tokens.txt", "r").read()

    keys = f_keys.split("\n")

    return keys[:4]

ckey, csecret, atoken, asecret = getKeys()


# print(s.sentiment("I’m like an hour fifteen into ONCE UPON A TIME... IN HOLLYWOOD and it’s beautiful and charming and absolutely nothing has happened yet"))

# print(s.sentiment("The movie is what, 3 hours? And the first 2 hours 45 min was a bit ramble-y but promised either an exciting convergence of the A and B plots, or an emotional reckoning of an old star being given a second chance. We got neither and in the last 15 minutes we got excessive gore as two grown men fight and kill teenagers. Which. I should have seen coming I GUESS."))