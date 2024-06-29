import os
import logging
from datetime import timedelta
from utils import initialize_logger, load_env_vars
from app_factory import get_app

logger = logging.getLogger()
load_env_vars()

KAFKA_INPUT_TOPIC = os.environ["input"]
KAFKA_OUTPUT_TOPIC = os.environ["output"]
USE_LOCAL_KAFKA = True if os.environ.get('use_local_kafka') is not None else False

WINDOW_SECONDS = 10

def get_timestamp(val: dict, *_): 
    return val["timestamp"]

def reduce_price(state: dict, val: dict) -> dict:
    state["last"] = val['price']
    state["min"] = min(state["min"], val['price']) 
    state["max"] = max(state["max"], val['price']) 

    return state

def init_reduce_price(val: dict) -> dict:    
    return {
        "last" : val['price'],
        "first": val['price'],
        "max": val['price'],
        "min": val['price'],
        "product_id": val['product_id'],
    }


def run(): 
    app = get_app(use_local_kafka=USE_LOCAL_KAFKA) 
    logger.info(f"Subscribing to topic {KAFKA_INPUT_TOPIC}")
    input_topic = app.topic(KAFKA_INPUT_TOPIC, value_deserializer="json") 
    logger.info(f"Producing to topic {KAFKA_OUTPUT_TOPIC}")
    output_topic = app.topic(KAFKA_OUTPUT_TOPIC, value_serializer="json") 
    sdf = app.dataframe(input_topic) 
    sdf = sdf.tumbling_window(timedelta(seconds=WINDOW_SECONDS), 0) \
        .reduce(reduce_price, init_reduce_price) \
        .final()
 
    sdf["open"] = sdf.apply(lambda v: v['value']['first'])
    sdf["high"] = sdf.apply(lambda v: v['value']['max'])
    sdf["low"] = sdf.apply(lambda v: v['value']['min'])
    sdf["close"] = sdf.apply(lambda v: v['value']['last'])
     
    sdf["product_id"] = sdf.apply(lambda v: v['value']['product_id'])
    sdf['timestamp'] = sdf.apply(lambda v: int(v['start']))
 
    sdf = sdf[['product_id', 'timestamp',
               'open', 'high', 'low', 'close',
               'start', 'end']]

    sdf = sdf.update(lambda val: print(f"Sending update: {val}"))
 
    sdf = sdf.to_topic(output_topic)
 
    app.run(sdf)

if __name__ == "__main__":
    initialize_logger(config_path="logging.yaml")
    run()