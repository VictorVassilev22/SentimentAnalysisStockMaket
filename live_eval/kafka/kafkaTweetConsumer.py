from datetime import datetime

from live_eval.live_classifier import predictLR
from kafka import KafkaConsumer
import config.props as props
import json
import config.auth as auth
import sys
import logging

# configure logger
logging.basicConfig(filename='live-errors.log', encoding='UTF-8', level=logging.WARN)
logger = logging.getLogger(__name__)

# Create kafka consumer, same conf as producer
consumer_tweets = KafkaConsumer(props.topics[0],
                                bootstrap_servers=props.bootstrap_servers,
                                value_deserializer=lambda x: json.loads(x.decode('utf-8')))


def get_retweeters(tweet):
    id = tweet['data']['id']
    return auth.tw_client.get_retweeters(id)


def get_author(tweet):
    tw_id = tweet['data']['id']
    return auth.tw_client.get_tweet(tw_id, expansions='author_id').data.author_id


sum_sentiments = 0
entry_count = 0

for message in consumer_tweets:
    try:
        sys.stdout.write('\r\r')
        tw = json.loads(json.dumps(message.value))
        text = tw['data']['text']
        sys.stdout.write(text + '\n')
        pred = predictLR(text)
        sum_sentiments += pred
        entry_count += 1
        sys.stdout.write('current: ' + str(pred) + '\t')
        sys.stdout.write('avg: ' + str(sum_sentiments / entry_count * 100) + '%')
    except Exception as e:
        error_message = f"Error ({datetime.now()}): {e}"
        logging.error(error_message)
        print(error_message)
        continue
