#!/bin/sh -xu

# flag outstanding work!
grep --colour 'TODO\|#' paper_outline.txt

# get started
cp paper_outline.txt paper.html
cp paper.html paper.tmp.html
perl -ple  's|^\s+||' paper.tmp.html > paper.html
#headers
#perl -ple  's|(h\d+):(.*?){|<$1>$2</$1>|' paper.txt >& paper.html
## adoptiong <p>
cp paper.html paper.tmp.html
perl -ple  's|(h\d+):(.*?){|<$1>$2</$1><pre>|' paper.tmp.html > paper.html

# remove single {}
cp paper.html paper.tmp.html
#perl -ple 's|^\s*[{}]\s*$||' paper_outline.txt  >& paper.txt
perl -ple 's|^\s*[}]\s*$|</pre>|' paper.tmp.html > paper.html

# references - TODO: generate normal text file, then number, then replace links in article with number
# e.g. <a href=url>[1]</a>
grep -o -P '\[htt.*?\]' paper_outline.txt | perl -ple 's|[\]\[]||g' | perl -ple 's|(.*)|<a href=$1>$1<a>|' >& references.html

