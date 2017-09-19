#pragma once

#include <vector>
#include "Kahan.h"
#include <fstream>
#include <numeric>
#include "Dataset.h"
#include <map>
using namespace std;

struct TDecisionTreeNode {
	int weight = 0;
	int featureIndex = 0;
	double threshold = 0;
	int classIndex = -1;
	int leftChildIndex = -1;
	int rightChildIndex = -1;
	static TDecisionTreeNode Train() {

	}
};

struct TDecisionTree {
	vector<TDecisionTreeNode> tree;
	template <typename T>
	int Prediction(const vector<T>& features) const {
		int idx = 0;
		while (tree[idx].classIndex < 0) {
			idx = features[tree[idx].featureIndex] < tree[idx].threshold ? tree[idx].leftChildIndex : tree[idx].rightChildIndex;
		}
		return tree[idx].classIndex;
	}
	void SaveToFile(ofstream& treeOut) {
		if (tree.size()) {
			treeOut << tree.size() << " ";
			for (TDecisionTreeNode node : tree) {
				treeOut << node.weight << " " << node.featureIndex << " " << node.threshold << " " << node.classIndex << " " << node.leftChildIndex << " " << node.rightChildIndex << " ";
			}
		}
	}
	static TDecisionTree LoadFromFile(ifstream& treeIn) {

		size_t treeSize;
		treeIn >> treeSize;


		TDecisionTree desicionTree;
		desicionTree.tree.resize(treeSize);

		for (size_t nodeIdx = 0; nodeIdx < treeSize; ++treeSize) {
			treeIn >> desicionTree.tree[nodeIdx].weight >> desicionTree.tree[nodeIdx].featureIndex >> desicionTree.tree[nodeIdx].threshold >> desicionTree.tree[nodeIdx].classIndex >> desicionTree.tree[nodeIdx].leftChildIndex >> desicionTree.tree[nodeIdx].rightChildIndex;
		}

		return desicionTree;
	};
	static TDecisionTree Train() {

	}
	vector<double> CalculateSplitpoints(TDataset& dataset, int featureIdx) {
		int instanceCount = dataset.featuresMatrix.size();

		vector<size_t> sortedIdx = dataset.SortByFeatureIdx(featureIdx);
		vector<double> splits;

		int i = 1;
		double lastClass = dataset.goals[sortedIdx[0]];
		double lastVal = dataset.featuresMatrix[sortedIdx[0]][featureIdx];
		while (i < instanceCount) {
			if (dataset.goals[sortedIdx[i]] == lastClass) {
				lastVal = dataset.featuresMatrix[sortedIdx[i]][featureIdx];
				++i;
				continue;
			}

			splits.push_back((lastVal + dataset.featuresMatrix[sortedIdx[i]][featureIdx]) / 2.);
			lastVal = dataset.featuresMatrix[sortedIdx[i]][featureIdx];
			lastClass = dataset.goals[sortedIdx[i]];
		}
		lastVal = (lastVal + dataset.featuresMatrix[sortedIdx[i-1]][featureIdx]) / 2.;
		if (lastVal != splits.back())
			splits.push_back(lastVal);
		return splits;
	}

	struct TBestSplit {
		int featureIdx;
		double splitVal;
	};

	TBestSplit FindBestSplit(TDataset& dataset) {
		int featureCount = dataset.featuresMatrix[0].size();
		TBestSplit bestSplit;
		bestSplit.featureIdx = -1;
		double minGini = 1;
		for (int featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
			vector<double> splitpoints = CalculateSplitpoints(dataset, featureIdx);			
			for (double split : splitpoints) {
				double Gini = CalculateGini(dataset, featureIdx, split);
				if (Gini < minGini) {
					minGini = Gini;
					bestSplit.featureIdx = featureIdx;
					bestSplit.splitVal = split;
				}
			}
		}
	}
	double CalculateGini(TDataset& dataset, int featureIdx, double split) {
		int instanceCount = dataset.featuresMatrix.size();
		map<double, int> classCounterLeft;
		map<double, int>::iterator itLeft;
		int sizeLeft = 0;
		map<double, int> classCounterRight;
		map<double, int>::iterator itRight;
		int sizeRight = 0;
		for (int i = 0; i < instanceCount; ++i) {
			if (dataset.featuresMatrix[i][featureIdx] < split) {
				itLeft = classCounterLeft.find(dataset.goals[i]);
				if (itLeft != classCounterLeft.end()) {
					++classCounterLeft[dataset.goals[i]];
				}
				else
					classCounterLeft.insert({ dataset.goals[i] , 1});
				++sizeLeft;
			}
			else {
				itRight = classCounterRight.find(dataset.goals[i]);
				if (itRight != classCounterRight.end()) {
					++classCounterRight[dataset.goals[i]];
				}
				else
					classCounterRight.insert({ dataset.goals[i] , 1 });
				++sizeRight;
			}
		}
		double GiniLeft = 1;
		double GiniRight = 1;
		if (sizeLeft) {
			for (itLeft = classCounterLeft.begin(); itLeft != classCounterLeft.end(); itLeft++) {
				double classProb = itLeft->second / sizeLeft;
				GiniLeft -= classProb*classProb;
			}
		}
		if (sizeRight) {
			for (itRight = classCounterRight.begin(); itRight != classCounterRight.end(); itRight++) {
				double classProb = itRight->second / sizeRight;
				GiniRight -= classProb*classProb;
			}
		}
		return (GiniLeft + GiniRight) / 2;
	}


};