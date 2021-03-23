cd $TESTDIR
unzip -o -q referenceResults.zip
cp ../compareResults.m compareResults.m
octave --no-gui compareResults.m
rm referenceResults.txt compareResults.m

