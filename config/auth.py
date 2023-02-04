import tweepy

# your authentication to Twitter API
api_key = 'REPLACE'
api_secret = 'REPLACE'
bearer_token = 'REPLACE'
access_token = 'REPLACE'
access_secret = 'REPLACE'


tw_client = tweepy.Client(bearer_token, api_key, api_secret, access_token, access_secret)
tw_auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
tw_api = tweepy.API(tw_auth)

