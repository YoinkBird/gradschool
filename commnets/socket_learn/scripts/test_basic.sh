#!/bin/sh -xu

scriptpath=`realpath $0`
scriptdir=`dirname $scriptpath`
#scriptdir=`dirname $0`
#repodir=`git rev-parse --show-toplevel`
#cd $repodir
$scriptdir/build_run.sh localhost 1025

