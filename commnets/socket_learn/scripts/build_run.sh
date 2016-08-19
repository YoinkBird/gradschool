#!/bin/sh -xu
scriptdir=`dirname $0`

echo $*

# compile
$scriptdir/build.sh
if [ $? -ne 0 ] ;then
  exit
fi

# now run
$scriptdir/run.sh $*
