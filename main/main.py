from quixstreams import Application
import os

app = Application(
    broker_address = os.environ["KAFKA_BROKER_ADDRESS"]
    consumer_group = "json__main_consumer_group"
)

input_topic = app.topic('trades', value_deserializer='json')
output_topic = app.topic('ohlc_features', value_serializer='json')

sdf = app.dataframe(input_topic)
sdf = sdf.tumbling_window(timedelta(seconds=WINDOW_SECONDS), 0).reduce(reduce_price, init_reduce_price).final()
sdf = sdf.to_topic(output_topic)

if __name__ = "__main__":
    app.run()