#!/bin/sh -xu
echo $*

# compile
javac EchoServer.java EchoClient.java
return $?
