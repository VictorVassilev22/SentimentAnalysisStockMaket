#!/bin/bash
cd "$KAFKA_HOME" || exit 1
KAFKA_CLUSTER_ID="$(./bin/kafka-storage random-uuid)" # generate cluster id
./bin/kafka-storage format -t "$KAFKA_CLUSTER_ID" -c ./etc/kafka/kraft/server.properties # format log directories
./bin/kafka-server-start ./etc/kafka/kraft/server.properties # start the kafka server
