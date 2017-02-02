#!/bin/bash -xu

scriptdir="$(dirname $0)"

cd $scriptdir
pwd
ls
source setup_gradle.sh
source setup_mingw.sh

scripts=$(ls ./setup_* | grep -v "setup_verify\|setup_all")


for script in ${scripts[@]}; do
  echo source $script
  source $script
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "-E- could not find $script"
    err+=1
    missingToolArr+=($script)
  fi
done

$SHELL
echo "exiting env setup"
