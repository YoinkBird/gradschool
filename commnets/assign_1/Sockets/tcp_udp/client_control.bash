#!/bin/bash -xu

port=$PORT
host=$MYHOST
user=$(date +%A%H_%M_%S)

action="list"
if [[ ! -z $* ]]; then
  if [[ ! -z ${1:-} ]]; then
    action=$1
  fi
fi

if [[ $action == "list" ]]; then
  lsof -i :${port}
  lsof -i :${port} | awk '{printf "%s\n",$2}' | grep -v 'PID' | xargs -r ps -f | cat | grep --colour '\bjava\b'
elif [[ $action == "hup" ]]; then
  lsof -i :${port} | awk '{printf "%s\n",$2}' | grep -v 'PID' | xargs -r kill -s HUP
elif [[ $action == "kill" ]]; then
  lsof -i :${port} | awk '{printf "%s\n",$2}' | grep -v 'PID' | xargs -r kill
elif [[ $action == "start" ]]; then
  #env USERPREFIX=`date +%A%H_%M_%S` env HOST_SERVER=192.168.10.225 HOST_PORT=1030 bash -c './java_build_run.sh $HOST_SERVER $HOST_PORT user${USERPREFIX}${i} & sleep 5; lsof -i :${HOST_PORT}'
  ./java_build_run.sh $host $port user${user} & sleep 5; lsof -i :${port}
fi


# start client with
# env USERPREFIX=defg env MYHOST=192.168.10.225 PORT=1030 bash -c './java_build_run.sh $MYHOST $PORT user${USERPREFIX}${i} & sleep 5; lsof -i :${PORT}'
