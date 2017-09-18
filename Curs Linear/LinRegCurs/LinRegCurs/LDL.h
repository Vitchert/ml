#pragma once
#include <vector>
#include "Kahan.h"
using namespace std;

namespace LDL {
	bool Decompose(const vector<double>& upperLinearizedMatrix,
		vector<vector<double>>& Lmatrix,
		vector<double>& Dmatrix,
		const double regularizationThreshold,
		const double regularizationParameter);
	void Decompose(const vector<double>& upperLinearizedMatrix,
		vector<vector<double>>& Lmatrix,
		vector<double>& Dmatrix);
	vector<double> SolveUpper(vector<double>& inputVector,
				const vector<vector<double>>& Lmatrix,
				const  vector<double>& Dmatrix);
	vector<double> SolveLover(vector<double> inputVector,
				const vector<vector<double>>& Lmatrix,
				const  vector<double>& Dmatrix);

	vector<double> solve(const vector<double>& upperLinearizedMatrix,
				vector<double>& Vector);
};