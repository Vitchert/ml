#pragma once

#include <vector>
#include "Kahan.h"
#include <fstream>
#include <numeric>
using namespace std;

struct TDecisionTreeNode {
	int weight = 0;
	int featureIndex = 0;
	double threshold = 0;
	int classIndex = -1;
	int leftChildIndex = -1;
	int rightChildIndex = -1;
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
};