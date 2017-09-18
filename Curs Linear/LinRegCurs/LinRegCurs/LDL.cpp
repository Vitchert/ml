#include "LDL.h"

bool LDL::Decompose(const vector<double>& upperLinearizedMatrix,
	vector<vector<double>>& Lmatrix,
	vector<double>& Dmatrix,
	const double regularizationThreshold,
	const double regularizationParameter)
{
	size_t size = Dmatrix.size();
	TKahan summator;
	vector<double> Lvector;
	size_t matrixIdx = 0;

	for (size_t j = 0; j < size; ++j) {
		Lvector.resize(size - j);

		summator = 0;
		for (size_t k = 0; k < j; ++k) {
			summator += Lmatrix[k][j - k] * Lmatrix[k][j - k] * Dmatrix[k];
		}
		Dmatrix[j] = upperLinearizedMatrix[matrixIdx++] - (double)summator + regularizationParameter;
		if (Dmatrix[j] < regularizationThreshold) {
			return false;
		}
		Lvector[0] = 1.;
		for (size_t i = j + 1; i < size; ++i) {
			summator = 0;
			for (size_t k = 0; k < j; ++k) {
				summator += Lmatrix[k][i - k] * Lmatrix[k][j - k] * Dmatrix[k];
			}
			Lvector[i - j] = ((upperLinearizedMatrix[matrixIdx++] - (double)summator) / Dmatrix[j]);
		}
		Lmatrix.push_back(Lvector);
	}
	return true;
}

void LDL::Decompose(const vector<double>& upperLinearizedMatrix,
	vector<vector<double>>& Lmatrix,
	vector<double>& Dmatrix)
{
	const double regularizationThreshold = 1e-5;
	double regularizationParameter = 0.;
	while (!Decompose(upperLinearizedMatrix,
		Lmatrix,
		Dmatrix,
		regularizationThreshold,
		regularizationParameter))
	{
		regularizationParameter = regularizationParameter ? 2 * regularizationParameter : 1e-5;
	}
}

vector<double> LDL::SolveUpper(vector<double>& inputVector, const vector<vector<double>>& Lmatrix, const  vector<double>& Dmatrix) {
	size_t size = inputVector.size();
	vector<double> xVector(size);
	TKahan summator;
	for (size_t i = size; i > 0; --i) {
		summator = 0;
		for (size_t k = size; k > i; --k) {
			summator += Dmatrix[i - 1] * Lmatrix[i - 1][k - i] * xVector[k - 1];
		}
		xVector[i - 1] = (inputVector[i - 1] - (double)summator) / (Dmatrix[i - 1] * Lmatrix[i - 1][0]);
	}
	return xVector;
}

vector<double> LDL::SolveLover(vector<double> inputVector, const vector<vector<double>>& Lmatrix, const  vector<double>& Dmatrix) {
	size_t size = inputVector.size();
	vector<double> yVector(size);
	TKahan summator;
	for (size_t i = 0; i < size; ++i) {
		summator = 0;
		for (size_t k = 0; k < i; ++k) {
			summator += Lmatrix[k][i - k] * yVector[k];
		}
		yVector[i] = (inputVector[i] - (double)summator);
	}
	return yVector;
}

vector<double> LDL::solve(const vector<double>& upperLinearizedMatrix, vector<double>& Vector) {
	vector<double> Dmatrix(Vector.size());
	vector<vector<double>> Lmatrix;
	Decompose(upperLinearizedMatrix, Lmatrix, Dmatrix);
	return SolveUpper(SolveLover(Vector, Lmatrix, Dmatrix), Lmatrix, Dmatrix);
}