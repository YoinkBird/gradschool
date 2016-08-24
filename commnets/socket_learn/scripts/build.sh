#!/bin/bash -xu

# common configs
scriptdir="$(dirname "$0")"
source "$scriptdir/config.sh"

# main section
echo $*

# compile
#javac ${javacOpts:-} $*
for file in "$@"; do
  #$javac_base_cmd ${javacOpts:-} $file
  $javac_base_cmd $file
done
exit $?
