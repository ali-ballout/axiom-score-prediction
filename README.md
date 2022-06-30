# axiom-score-prediction
Code to replicate the Experiments in the paper submitted to WI 2022


1.Available are all files needed to perform a full evaluation.to do so you need to install corese server.
	a. download and install the latest corese server from inria 
	b. place the files in the same directory where you install corese for ease of use
  C.place the dbpedia owl file included in this repo in the directory of corese

2.The file main.py, contains the code that produces the proposed similarity measure can be easily run with one of the two dataset files included which are the subclass of and the disjointwith scored axioms. No need to edit anything except the name of the file in the main function, everything is commented for ease of use.

3.The "instance based similarity.R" is the code used to create the instace based similarity measure, it has a specific input folder named accordingly and already written in the code for ease of use. A package for sparql is included in the repo as it is no longer provided as a package for R. If you run this code it will take about a week to finish.

4.An included file named "model testing.py" includes the code needed to read all the produced (and included data sets), with defined models, hyper parameter disctionaries for each model to allow the user to run experiments and replicate the results of the paper.
