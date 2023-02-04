#!/bin/bash

die () {
    echo >&2 "$@"
    exit 1
}

[ "$#" -eq 4 ] || die "Required 4 args: port, replication factor, partitions, topic name"

cd "$KAFKA_HOME" || die "change directory failed"

./bin/kafka-topics --bootstrap-server localhost:"$1" \
                   --create \
                   --replication-factor "$2" \
                   --partitions "$3" \
                   --topic "$4"