#!/bin/bash -xu

main(){
  # common configs
  scriptdir="$(dirname "$0")"
  source "$scriptdir/config.sh"

  if [ $# -ne 0 ]; then
    testConfig=$1;
  else
    testConfig="test/cfg_basic.sh"
  fi
  export THIS_TEST_CFG=$testConfig
  $scriptdir/build_run.sh
}

main "$@"
