#!/bin/bash -xu

# create jar:
# http://docs.oracle.com/javase/tutorial/deployment/jar/build.html
# http://docs.oracle.com/javase/tutorial/deployment/jar/downman.html
# Client
# java -jar Chatter.jar <screen_name> <MemD_server_hostname> <MemD_welcome_tcp_port>
# java Chatter <screen_name> <MemD_server_hostname> <MemD_welcome_tcp_port>
# Server
# java -jar MemD.jar <MemD_welcome_tcp_port>
# or
# java MemD.java <MemD_welcome_tcp_port>

# prepare dir
mkdir -p deliver
mkdir -p deliver/Chatter/tcp_udp
mkdir -p deliver/MemD/tcp_udp
tree deliver

# distributed files
ls SimpleGui/basic/Gui.java #,tcp_udp/Client.java}
ls Sockets/concurrentTCP/Server.java #,tcp_udp/Protocol.java}
ls Sockets/tcp_udp/{Client.java,Protocol.java}

# consolidate, organise files
cp SimpleGui/basic/Gui.java deliver/Chatter
cp Sockets/concurrentTCP/Server.java deliver/MemD
cp Sockets/tcp_udp/{Client.java,Protocol.java} deliver/Chatter/tcp_udp/
cp Sockets/tcp_udp/{Client.java,Protocol.java} deliver/MemD/tcp_udp/

# prepare files
cd deliver/Chatter
sed -i 's|package.*||g' Gui.java 
cd -

cd deliver/MemD
sed -i 's|package.*||g' Server.java
cd -

# create jar
cd deliver/Chatter
javac -cp . Gui.java tcp_udp/*
echo "Main-Class: Gui" > manifest.txt
jar cfvm Chatter.jar manifest.txt *
cd -

cd deliver/MemD
javac -cp . Server.java tcp_udp/*
echo "Main-Class: Server" > manifest.txt
jar cfvm MemD.jar manifest.txt *
cd -

mv deliver/Chatter/Chatter.jar deliver
mv deliver/MemD/MemD.jar deliver

tree deliver

# run
cd deliver
java -jar MemD.jar 1028 &
pid=$!
sleep 2

java -jar Chatter.jar localhost 1028 orzo &
java -jar Chatter.jar localhost 1028 zoro
cd -

# end
PORT=1028
MYHOST=localhost 
lsof -i :${PORT} | awk '{printf "%s\n",$2}' | grep -v 'PID' | xargs ps -f | cat | awk '{printf "%s\n",$2}' | grep -v 'PID' | xargs -r kill
kill $pid

exit;
