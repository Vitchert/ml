#pragma once

#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "DecisionTree.h"

struct TRForestModel{
	int classesCount = 8;
	std::vector<TDecisionTree> forest;	

	template <typename T>
	int Prediction(const std::vector<T>& features) const {
		int prediction = -1;
		int maxIdx = -1;
		int max = 0;
		std::vector<int> results(classesCount, 0);

		for (TDecisionTree tree : forest) {		
			prediction = tree.Prediction(features);
			++results[prediction];
			if (results[prediction] > max) {
				max = results[prediction];
				maxIdx = prediction;
			}
		}
		
		return maxIdx;
	}
	void SaveToFile(const std::string& modelPath) {
		ofstream modelOut(modelPath);
		modelOut.precision(20);

		modelOut << classesCount << " " << forest.size() << " ";

		for (TDecisionTree tree : forest) {
			tree.SaveToFile(modelOut);
		}
	}
	static TRForestModel LoadFromFile(const std::string& modelPath) {
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