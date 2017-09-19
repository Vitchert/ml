#pragma once

#include "RForestModel.h"
#include "Dataset.h"

class TRForestSolver {
private:
	TDataset& dataset;

public:
	void AddDataset(const TDataset& data) {
		dataset = data;
	}

	TRForestModel Solve(const int& treeCount) {
		TDataset::TBaggingIterator iterator = dataset.BaggingIterator();
		TRForestModel model;
		dataset.sortByFeature();
		for (int i = 0; i < treeCount; ++i) {
			TDecisionTree dTree;
			iterator.ResetShuffle(i);
			iterator.SetLearnMode();






			model.forest.push_back(dTree);
		}
	}

};