#!/bin/sh -u
repodir=`git rev-parse --show-toplevel`
# globals
# optional: write class files to a specific dir
javacOpts=""
javaOpts=""
if [[ ! -z ${THIS_JAVAC_DESTDIR:-} ]]; then
  destDir=$THIS_JAVAC_DESTDIR
  mkdir -p $destDir
  javacOpts="-d $destDir"
  javaOpts="-cp $destDir"
fi

javac_base_cmd="javac"
java_base_cmd="java"
