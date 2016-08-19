#!/bin/bash -xu
echo $*
echo $1
echo $2
host=$1
port=$2

java EchoServer $port &
sleep 1
lsof -i :${port}

# named pipe
# http://serverfault.com/a/297095 
inputPipe="/tmp/javaEchoClient_input"
mkfifo /tmp/javaEchoClient_input
cat > /tmp/javaEchoClient_input &
echo $! > /tmp/javaEchoClient_input-cat-pid
cat /tmp/javaEchoClient_input | java EchoClient $host $port &

for i in {0..5}; do 
  testVal=test${i};
  echo $testVal > $inputPipe
done
echo "TERMINATE" > $inputPipe
# resolves buffer issues in which java output appears after everything is done
sleep 2
jobs
echo "done"
