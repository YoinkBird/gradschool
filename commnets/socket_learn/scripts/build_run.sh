#!/bin/sh -xu
scriptdir=`dirname $0`
scriptdir="$(dirname "$0")"

echo $*

java=$THIS_JAVAFILES
# compile
$scriptdir/build.sh $java
if [ $? -ne 0 ] ;then
  exit
fi

# now run
$scriptdir/run.sh $*
