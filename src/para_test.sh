#! /bin/bash

if [[ -z $1 ]] || [[ ! -f $1 ]];then
    echo "Usage: `basename $0` <corpus_filename>"
    exit 1
fi

for core in {1..4};do
    for i in {1..5};do
        ./phrase_suggester analyze $1 \
        +RTS -s -ls -H4G -N$core 2>&1 \
        | grep "Total" \
        | awk -v pre="[c=$core, i=$i] " '$0=pre $0'
        sleep 5
    done
done