#pragma once

#include <vector>
#include "Kahan.h"
#include <fstream>
#include <numeric>
#include <algorithm>
#include "DecisionTree.h"

using namespace std;

struct TRForestModel{
	int classesCount;
	vector<TDecisionTree> forest;	

	template <typename T>
	int Prediction(const vector<T>& features) const {
		int prediction = -1;
		int maxIdx = -1;
		int max = 0;
		vector<int> results(classesCount, 0);

		for (TDecisionTree tree : forest) {		
			prediction = tree.Prediction(features);
			++results[prediction];
			if (results[prediction] > max) {
				max = results[prediction];
				maxIdx = prediction;
			}
		}
		
		return maxInd;
	}
	void SaveToFile(const string& modelPath) {
		ofstream modelOut(modelPath);
		modelOut.precision(20);

		modelOut << classesCount << " " << forest.size() << " ";

		for (TDecisionTree tree : forest) {
			tree.SaveToFile(modelOut);
		}
	}
	static TRForestModel LoadFromFile(const string& modelPath) {
		ifstream modelIn(modelPath);
		size_t classCount;
		size_t forestSize;
		modelIn >> classCount >> forestSize;

		TRForestModel model;
		model.classesCount = classCount;
		for (int i = 0; i < forestSize; ++i) {
			model.forest[i] = TDecisionTree::LoadFromFile(modelIn);
		}
		return model;
	};

};