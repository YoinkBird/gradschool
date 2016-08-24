#!/bin/bash -xu

# common configs
scriptdir="$(dirname "$0")"
source "$scriptdir/config.sh"

# main section
echo $*

# compile
javac_base_cmd ${javacOpts:-} $*
exit $?
