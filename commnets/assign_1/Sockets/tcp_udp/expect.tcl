#!/usr/bin/expect -f
# super hacky expect script
# launch a client1 , then launch client2 to send the hack 'exit' sequence', then check that client1 exits
# https://spin.atomicobject.com/2016/01/11/command-line-interface-testing-tools/
puts "################################################################################"
puts "starting client1"
spawn ./java_build_run.sh
set spawn_id_client1 $spawn_id
expect "waiting"
# expect '*getUdp*: waiting * reply on *'
puts "################################################################################"
puts "starting client2"
spawn ./java_build_run.sh localhost 1024 fettucini 'EXIT:tagliatelle'
set spawn_id_client2 $spawn_id
puts "################################################################################"
puts "back to client1"
set spawn_id $spawn_id_client1
#expect "received"
expect "received EXIT"
puts "\n"
puts "################################################################################"
puts "\n"

