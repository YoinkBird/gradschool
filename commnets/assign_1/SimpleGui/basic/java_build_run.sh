#!/bin/bash -xu

args=$*
if [[ -z $args ]]; then
  args="localhost 1024 orzo"
fi


# clean compile with one-shot backup
rm -rf cls.bak
mv cls cls.bak
mkdir -p cls
javac -d cls -cp . Gui.java tcp_udp/Client.java 
if [[ $? -ne 0 ]]; then
  exit $?
fi

java -cp cls basic.Gui $args
