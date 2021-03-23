X = load("results.txt");
Y = load("referenceResults.txt");
err = norm(X-Y,"columns")./norm(Y,"columns");
save("-ascii","error.txt","err")
