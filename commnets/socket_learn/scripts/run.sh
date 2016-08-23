#!/bin/bash -xu
echo $*
echo $1
echo $2
host=$1
port=$2

java EchoServer $port &
java_cmd_pid=$!
sleep 1
lsof -i :${port}

cmd_client="java EchoClient $host $port"

usepipe=1
if [[ $usepipe -eq 1 ]];then
  # named pipe
  # http://serverfault.com/a/297095 
  inputPipe="/tmp/javaEchoClient_input"
  mkfifo /tmp/javaEchoClient_input
  cat > /tmp/javaEchoClient_input &
  cat_cmd_pid=$!
  echo $cat_cmd_pid > /tmp/javaEchoClient_input-cat-pid
  cat /tmp/javaEchoClient_input | java EchoClient $host $port &

  for i in {0..5}; do
    testVal=test${i};
    echo $testVal > $inputPipe
  done
  echo "TERMINATE" > $inputPipe
  # resolves buffer issues in which java output appears after everything is done
else
  for i in {0..5}; do
    testVal=test${i};
    jobs
    echo $testVal | $cmd_client
  done
  echo "TERMINATE" | $cmd_client

fi
sleep 2
jobs
kill -9 $java_cmd_pid
kill -9 $cat_cmd_pid
sleep 2
jobs
echo "done"
