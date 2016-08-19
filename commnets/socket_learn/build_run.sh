#!/bin/sh -xu
echo $*

# compile
javac EchoServer.java EchoClient.java
if [ $? -ne 0 ] ;then
  exit
fi

# now run
./run.sh $*
