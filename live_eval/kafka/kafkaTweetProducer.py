import config.auth as auth
import tweepy
import logging
from kafka import KafkaProducer
import config.props as props


# We overwrite our tweet listener in order to send info to our kafka consumer
class TweetStream(tweepy.StreamingClient):

    def on_connect(self):
        print("Connected!")
        self.cleanup()
        self.add_rules(props.search_rules)

    def on_data(self, raw_data):
        logger.info(raw_data)
        producer.send(props.topics[0], value=raw_data)
        return True

    def cleanup(self):
        rule_ids = []
        rules = self.get_rules()

        if rules.data is not None:
            for rule in rules.data:
                rule_ids.append(rule.id)

        if len(rule_ids) > 0:
            self.delete_rules(rule_ids)


if __name__ == '__main__':
    # Generate Kafka producer / localhost and 9092 default ports
    producer = KafkaProducer(bootstrap_servers=props.bootstrap_servers)
    # Create logging instance
    logging.basicConfig(filename='debug.log', encoding='UTF-8', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Twitter API usage
    twitter_stream = TweetStream(auth.bearer_token)
    # twitter_stream.cleanup()

    # twitter_stream.add_rules(props.search_rules)
    twitter_stream.filter()
