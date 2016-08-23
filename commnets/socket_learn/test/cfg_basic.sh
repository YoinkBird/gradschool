#!/bin/sh -xu

scriptpath=`realpath $0`
scriptdir=`dirname $scriptpath`
#scriptdir=`dirname $0`
this_server="EchoServer"
this_client="EchoClient"
export THIS_SERVER=$this_server
export THIS_CLIENT=$this_client
javaFiles="${this_server}.java ${this_client}.java"
export THIS_JAVAFILES="$javaFiles"
export THIS_CLI_OPTS="localhost 1025"

