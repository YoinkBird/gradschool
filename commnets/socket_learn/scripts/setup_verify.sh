#!/bin/sh -xu

which javac
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "-E- could not find javac, exiting"
  exit $rc
fi
