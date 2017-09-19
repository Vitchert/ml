#pragma once

#include "RForestModel.h"
#include "Dataset.h"

class TRForestSolver {
private:
	TDataset dataset;
	vector<vector<size_t>> sortedFeaturesIdxMatrix;
	vector<vector<double>> splitPoints;
public:
	void AddDataset(const TDataset& data) {
		dataset = data;
		sortedFeaturesIdxMatrix = dataset.sortByFeature();
		CalculateSplitpoints();

	}
	void Add(const vector<double>& features, const double goal, const double weight = 1.) {
		dataset.featuresMatrix.push_back(features);
		dataset.goals.push_back(goal);
		dataset.weights.push_back(weight);
	}
	void CalculateSplitpoints() {
		int featureCount = sortedFeaturesIdxMatrix.size();
		int instanceCount = sortedFeaturesIdxMatrix[0].size();
		
		for (int j = 0; j < featureCount; ++j) {
			vector<double> splits;
			int i = 1;
			double lastClass = dataset.goals[sortedFeaturesIdxMatrix[j][0]];
			double lastVal = dataset.featuresMatrix[sortedFeaturesIdxMatrix[j][0]][j];
			while (i < instanceCount) {
				if (dataset.goals[sortedFeaturesIdxMatrix[j][i]] == lastClass) {
					lastVal = dataset.featuresMatrix[sortedFeaturesIdxMatrix[j][i]][j];
					++i;
					continue;
				}

				splits.push_back((lastVal + dataset.featuresMatrix[sortedFeaturesIdxMatrix[j][i]][j])/2.);
				lastVal = dataset.featuresMatrix[sortedFeaturesIdxMatrix[j][i]][j];
				lastClass = dataset.goals[sortedFeaturesIdxMatrix[j][i]];
			}
			lastVal = (lastVal + dataset.featuresMatrix[sortedFeaturesIdxMatrix[j][i]][j]) / 2.;
			if (lastVal != splits.back())
				splits.push_back(lastVal);
			splitPoints.push_back(splits);
			splits.clear();
		}
	}

	TRForestModel Solve(const int& treeCount) {
		TDataset::TBaggingIterator iterator = dataset.BaggingIterator();
		TRForestModel model;
		

		for (int i = 0; i < treeCount; ++i) {
			TDecisionTree dTree;
			iterator.ResetShuffle(i);
			iterator.SetLearnMode();
			dTree = ConstructTree();
			model.forest.push_back(dTree);		
			//TODO OOB
		}
		return model;
	}

	

};