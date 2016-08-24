#!/bin/sh -xu

# build: 
gradle -q --build-file build_server.gradle build
# run:
gradle -q --build-file build_server.gradle run
