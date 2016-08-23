# optional: write class files to a specific dir
export THIS_JAVAC_DESTDIR="bin"
javacOpts=""
javaOpts=""
if [[ ! -z ${THIS_JAVAC_DESTDIR:-} ]]; then
  destDir=$THIS_JAVAC_DESTDIR
  mkdir -p $destDir
  javacOpts="-d $destDir"
  javaOpts="-cp $destDir"
fi

