#pragma once

#include "LDL.h"
#include "LinearModel.h"
#include "Dataset.h"

class TWelfordSolver {
private:

	vector<double> featureMeans;
	double goalsMean;
	vector<double> upperLinearizedOLSMatrix; // [firstStr, SecondStr, ...]
	vector<double> OLSVector;
	TKahan sumWeights;

	void CalculateCovariation();

public:
	TWelfordSolver() : goalsMean(0) {};
	void Add(const vector<double>& features, const double goal, const double weight = 1.);
	void AddDataset(const TDataset& dataset);
	TLinearModel Solve();

};