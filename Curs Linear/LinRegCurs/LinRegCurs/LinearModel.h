#pragma once

#include <vector>
#include "Kahan.h"
#include <fstream>
#include <numeric>
using namespace std;

struct TLinearModel {
	vector<double> Coefficients;
	double Intercept;

	template <typename T>
	double Prediction(const vector<T>& features) const {
		return inner_product(Coefficients.begin(), Coefficients.end(), features.begin(), Intercept);
	}
	void SaveToFile(const string& modelPath) {
		ofstream modelOut(modelPath);
		modelOut.precision(20);

		modelOut << Coefficients.size() << " ";
		modelOut << Intercept << " ";

		for (const double coefficient : Coefficients) {
			modelOut << coefficient << " ";
		}
	}
	void PredictionError(const vector<vector<double>>& featuresMatrix,
				const vector<double>& goals,
				const vector<double>& weights,
				const string& predictionsPath) 
	{
		size_t predictionsCount = goals.size();
		vector<double> predictions(predictionsCount);

		ofstream predictionsOut(predictionsPath);
		predictionsOut.precision(20);

		for (size_t predictionIdx = 0; predictionIdx < predictionsCount; ++predictionIdx) {
			predictions[predictionIdx] = Prediction(featuresMatrix[predictionIdx]);
		}

		predictionsOut << SumSquaredError(predictions, goals, weights) << "\n";

		for (size_t predictionIdx = 0; predictionIdx < predictionsCount; ++predictionIdx) {
			predictionsOut << predictions[predictionIdx] << "\n";
		}
	}
	double SumSquaredError(const vector<double>& predictions, const vector<double>& goals, const vector<double>& weights) {
		TKahan summ;
		size_t predictionsCount = goals.size();

		for (size_t predictionIdx = 0; predictionIdx < predictionsCount; ++predictionIdx) {
			summ += weights[predictionIdx] * (predictions[predictionIdx] - goals[predictionIdx])*(predictions[predictionIdx] - goals[predictionIdx]);
		}
		return summ;
	}
	static TLinearModel LoadFromFile(const string& modelPath) {
		ifstream modelIn(modelPath);

		size_t featuresCount;
		modelIn >> featuresCount;

		TLinearModel model;
		model.Coefficients.resize(featuresCount);

		modelIn >> model.Intercept;
		for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
			modelIn >> model.Coefficients[featureIdx];
		}

		return model;
	};
};