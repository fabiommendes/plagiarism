Library for detecting plagiarism in source code. Userful for online judges, 
teachers, developers and maybe lawyers.

Plagiarism uses the method described [here](http://...). The basic idea is to 
classify each submitted file according to different metrics and perform a 
series of k-means based clusterizations to determine which objects are most 
similar to each other. This approach has a N log N cost and scales fairly well 
to big samples.

The algorithm can be applied to natural text, source code and can even be 
adapted to run on arbitrary data structures (such as the parse tree of a 
computer program, ASM output, even binary executables). It requires some tuning
for each application and accuracy may vary widely depending on application.
You should expect better results grading Python and C source code. Performance
on other programming languages or even in other domains may vary.    
