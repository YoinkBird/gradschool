#!/bin/bash -xu
set +x
echo "# for diff : create reference as output.txt.`git log -n 1 --oneline HEAD | awk '{printf "%s",$1}'`"
echo "# for diff : redirect output to output.txt.`date --iso-8601=seconds`"
set -x

args=$*
if [[ -z $args ]]; then
  args="localhost 1024 farfalle"
fi


../../../../bin/java_build_run.sh ClientWalla.java localhost 1024 tagliatelle
exit;

# clean compile with one-shot backup
rm -rf cls.bak
mv cls cls.bak
mkdir -p cls
javac -d cls -cp . Gui.java tcp_udp/Client.java 
if [[ $? -ne 0 ]]; then
  exit $?
fi

java -cp cls basic.Gui $args
