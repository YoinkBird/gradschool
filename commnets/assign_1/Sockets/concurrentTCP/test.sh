#!/bin/sh -xu

kill_port_procs(){
  port=${1:-}
  lsof -i :${port}
  lsof -i :${port} | awk '{printf "%s\n",$2}' | grep -v PID | xargs -r kill
}

echo "################################################################################"
echo "starting client1"
port=1028
./java_build_run.sh &
rc=$?
java_cmd_pid=$!
# exit if bad compile
if [ $rc -ne 0 ] ;then
  kill $java_cmd_pid
  kill_port_procs $port
  exit $rc
fi
sleep 1
lsof -i :${port}

sleep 2;

#echo "################################################################################"
#echo "starting client2"
../tcp_udp/java_build_run.sh localhost $port fettucini 'EXIT:tagliatelle' > /dev/null 2>&1
#rc=$?
#echo "client2 rc: $rc"
#echo "################################################################################"
#echo "back to client1"

fg $java_cmd_pid
#kill -9 $java_cmd_pid
kill_port_procs $port

