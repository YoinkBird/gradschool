#!/bin/sh -u

err=0
missingToolArr=("")
tools=(
"javac"
"gradle"
)
for tool in ${tools[@]}; do
  which $tool
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "-E- could not find $tool"
    err+=1
    missingToolArr+=($tool)
  fi
done

echo "$err tools not found"
for tool in ${missingToolArr[@]}; do
  echo $tool
done
