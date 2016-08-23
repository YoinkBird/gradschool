#!/bin/bash -xu

# common configs
scriptdir="$(dirname "$0")"
source "$scriptdir/config.sh"

# test config
source "$scriptdir/test_basic.sh"

# main section
echo $*

java=$THIS_JAVAFILES
# compile
$scriptdir/build.sh $java
if [ $? -ne 0 ] ;then
  exit
fi

# now run
$scriptdir/run.sh $THIS_CLI_OPTS
