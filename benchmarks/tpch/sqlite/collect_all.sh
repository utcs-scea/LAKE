#!/bin/bash

#rm -f tpc.db
#cat sqls/all.sql | sqlite3 tpc.db

for i in $(seq 1 22); do 
    LD_PRELOAD="/home/hfingler/hf-HACK/toys/io_tracer/shim.so" sqlite3 tpc.db < queries/$i.sql 2>trace_$i.trace
done