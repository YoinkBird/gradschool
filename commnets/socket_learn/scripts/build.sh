#!/bin/bash -xu
echo $*

# options
scriptdir="$(dirname "$0")"
source "$scriptdir/config.sh"
# compile
javac ${javacOpts:-} $*
exit $?
