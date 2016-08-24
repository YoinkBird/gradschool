#!/bin/bash -u
scriptName="${0##*/}"
echo_usage()
{
  cat << EOF
  Usage: cd to dir with .java file to be compiled and run this script
  0 args: script assumes main classfile is same as basename \$PWD
    e.g. $scriptName
  1 args: .java file - tell script which file to run on
    e.g. $scriptName Simulator
  2 args: .java file, java args - args to .java file
    e.g. $scriptName Simulator 2,1

EOF
}
extract_package(){
  filePath=$1
#  package=`grep -P '^\s*package\s+.*?;' $curDir/${fileName}| sed 's|package \(.*\);|\1|g'`
  package=`perl -nle 'if(m|^\s*package\s+(.*?);|g){print $1}' $filePath`
  # http://stackoverflow.com/questions/13589895/shell-command-to-strip-out-m-characters-from-text-file
  package=`echo $package | sed 's|||g'`
  echo $package
}

main(){
  # print usage
  echo_usage
  #args=("$@")
  set -x
  echo $@
  # get filename, classpath based on pwd and package
  curDir=`pwd`;
  fileRoot=`basename $curDir`;
  mainClassFile=`grep -Pl '\bmain\b\s*\(String.*?args.*?\)' $curDir/*`;
  mainClassFile=`basename $mainClassFile`;
  mainClassFileRoot=`echo $mainClassFile | perl -nle 'if(m|(.*?).java|g){print $1}'`
  if [ $# -ne 0 ]; then
    fileRoot=$1;
  fi
  if [ $# -gt 1 ]; then
    targetArgs=$2;
  fi
  #TODO: check that it is a file and not a dir
  if [ ! -r $fileRoot ]; then
    fileName=${fileRoot}.java;
  fi
  # if dir-based name detection doesn't work, try the experimental 'grep main' detection
  if [ ! -r $fileName ]; then
    fileName=$mainClassFile
    fileRoot=$mainClassFileRoot
  fi
  # error if file not found
  if [ ! -r $fileName ]; then
    echo "-E- file not found: $fileName";
    exit 1;
  fi
  # fn's are basically processes ( http://stackoverflow.com/a/17338371 )
  # and therefore have to return non-status values by echo'ing http://stackoverflow.com/a/17336953
  package=$(extract_package "$curDir/$fileName")
  echo $package
  packagePath=''
  target=${fileRoot};
  if [ ! -z $package ]; then
    packagePath=`echo $curDir | sed "s|/$package||g"`
    target=${package}.${target}
  fi
  #className=`grep "public class" $curDir/${fileRoot}.java | sed 's|public class \(.*\){|\1|g'`
  export CLASSPATH=$curDir:$packagePath;
  export BUILDFILE=$fileRoot;
  #sh -cx  'buildfile=${BUILDFILE}; javac -cp .:${CLASSPATH} ${buildfile}.java && java -cp . ${buildfile}'
  #javac -cp .:${CLASSPATH} ${buildfile}.java

  # script is named build_run and has two symlinks: build run
  # three options:
  # build_run javac java
  # build     javac
  # run             java
  if [[ "$scriptName" =~ build  ]]; then
    mkdir -p class
    javac -d class ${fileRoot}.java
    rc=$?
    # exit if bad compile
    if [ $rc -ne 0 ] ;then
      exit $rc
    fi
  fi
  if [[ "$scriptName" =~ run ]]; then
    java -cp class ${target} ${targetArgs}
    rc=$?
    # exit if bad run
    if [ $rc -ne 0 ] ;then
      exit $rc
    fi
  fi
  exit $rc

}
main "$@"
