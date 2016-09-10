#!/bin/bash -xu

kill_port_procs(){
  port=${1:-}
  lsof -i :${port}
  lsof -i :${port} | awk '{printf "%s\n",$2}' | grep -v PID | xargs -r kill
}

java_cmd_pid_arr=();
echo "################################################################################"
echo "starting client1"
port=1028
./java_build_run.sh &
java_cmd_pid1=$!
sleep 1
lsof -i :${port}

sleep 2;

#echo "################################################################################"
#echo "starting client2"
waittime=3
../tcp_udp/java_build_run.sh localhost $port fettucini 'hello1' > /dev/null 2>&1 &
java_cmd_pid_arr+=("$!")
java_cmd_pid2=$!
sleep $waittime;
../tcp_udp/java_build_run.sh localhost $port fettucini 'hello2' > /dev/null 2>&1 &
java_cmd_pid_arr+=("$!")
java_cmd_pid3=$!
sleep $waittime;
../tcp_udp/java_build_run.sh localhost $port orzo      'hello3' > /dev/null 2>&1 &
java_cmd_pid_arr+=("$!")
sleep $waittime;
../tcp_udp/java_build_run.sh localhost $port arrancini 'hello3' > /dev/null 2>&1 &
java_cmd_pid_arr+=("$!")
sleep $waittime;
#rc=$?
#echo "client2 rc: $rc"
#echo "################################################################################"
#echo "back to client1"

jobs
fg $java_cmd_pid1
kill $java_cmd_pid1 $java_cmd_pid2 $java_cmd_pid3
for pid in ${java_cmd_pid_arr[@]}; do
  kill $pid
done
#kill -9 $java_cmd_pid
kill_port_procs $port

