#include "Welford.h"

void TWelfordSolver::CalculateCovariation() {
	size_t featuresCount = OLSVector.size();
	vector<double>::iterator olsMatrixElement = upperLinearizedOLSMatrix.begin();
	vector<double>::iterator olsVectorElement = OLSVector.begin();

	for (size_t featureNumber = 0; olsVectorElement != OLSVector.end(); ++featureNumber) {
		*olsVectorElement++ = sumWeights * (*olsVectorElement - goalsMean * featureMeans[featureNumber]);

		for (size_t featureNumber2 = featureNumber; featureNumber2 < featuresCount; ++featureNumber2) {
			*olsMatrixElement++ = sumWeights * (*olsMatrixElement - featureMeans[featureNumber] * featureMeans[featureNumber2]);
		}
	}
}

void TWelfordSolver::Add(const vector<double>& features, const double goal, const double weight) {
	const size_t featuresCount = features.size();

	if (featureMeans.empty()) {
		featureMeans.resize(featuresCount);
		OLSVector.resize(featuresCount);
		upperLinearizedOLSMatrix.resize(featuresCount * (featuresCount + 1) / 2);
	}

	sumWeights += weight;

	goalsMean += weight * (goal - goalsMean) / sumWeights;   //Update (static sumWeights) in TMeanCalculatorWelford

	vector<double>::iterator olsMatrixElement = upperLinearizedOLSMatrix.begin();
	for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
		featureMeans[featureNumber] += weight * (features[featureNumber] - featureMeans[featureNumber]) / sumWeights;
		OLSVector[featureNumber] += weight * (features[featureNumber] * goal - OLSVector[featureNumber]) / sumWeights;//Computes Sum_k(X_{k,i} * Y_i) / sumWeights

		for (size_t featureNumber2 = featureNumber; featureNumber2 < featuresCount; ++featureNumber2) {
			*olsMatrixElement += weight * (features[featureNumber] * features[featureNumber2] - *olsMatrixElement) / sumWeights;//Computes Sum_k(X_{k,i} * X_{k,j}) / sumWeights
			++olsMatrixElement;// else: vector iterator not dereferencable
		}
	}
};

void TWelfordSolver::AddDataset(const TDataset& dataset) {
	size_t datasetSize = dataset.goals.size();

	for (size_t i = 0; i < datasetSize; ++i) {
		Add(dataset.featuresMatrix[i], dataset.goals[i], dataset.weights[i]);
	}
};

TLinearModel TWelfordSolver::Solve() {
	TLinearModel model;
	CalculateCovariation();

	model.Coefficients = LDL::solve(upperLinearizedOLSMatrix, OLSVector);
	model.Intercept = goalsMean;
	const size_t featuresCount = OLSVector.size();
	for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
		model.Intercept -= featureMeans[featureNumber] * model.Coefficients[featureNumber];
	}

	return model;
}
