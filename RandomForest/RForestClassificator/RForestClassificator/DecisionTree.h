#pragma once
#include <vector>
#include <fstream>
#include <numeric>
#include <map>
#include <chrono> 
#include "Dataset.h"

#define EACH_NODE_SHUFFLE
struct TDecisionTreeNode {
	int weight = 0;
	int featureIndex = 0;
	double threshold = 0;
	int classIndex = -1;
	int leftChildIndex = -1;
	int rightChildIndex = -1;
};

struct TDecisionTree {
	std::vector<TDecisionTreeNode> tree;
	template <typename T>
	int Prediction(const std::vector<T>& features) const {
		int idx = 0;
		while (tree[idx].classIndex < 0) {
			idx = features[tree[idx].featureIndex] < tree[idx].threshold ? tree[idx].leftChildIndex : tree[idx].rightChildIndex;
		}
		return tree[idx].classIndex;
	}
	void SaveToFile(std::ofstream& treeOut) {
		if (tree.size()) {
			treeOut << tree.size() << " ";
			for (TDecisionTreeNode node : tree) {
				treeOut << node.weight << " " << node.featureIndex << " " << node.threshold << " " << node.classIndex << " " << node.leftChildIndex << " " << node.rightChildIndex << " ";
			}
		}
	}
	static TDecisionTree LoadFromFile(std::ifstream& treeIn) {

		size_t treeSize;
		treeIn >> treeSize;


		TDecisionTree desicionTree;
		desicionTree.tree.resize(treeSize);

		for (size_t nodeIdx = 0; nodeIdx < treeSize; ++treeSize) {
			treeIn >> desicionTree.tree[nodeIdx].weight >> desicionTree.tree[nodeIdx].featureIndex >> desicionTree.tree[nodeIdx].threshold >> desicionTree.tree[nodeIdx].classIndex >> desicionTree.tree[nodeIdx].leftChildIndex >> desicionTree.tree[nodeIdx].rightChildIndex;
		}

		return desicionTree;
	};

	void ConstructTree(TDataset& dataset, std::vector<char>& dataIdx, std::vector<int> classCount,int dataIdxsize, std::vector<int> fIdx) {
		TDecisionTreeNode node;
		TBestSplit split = FindBestSplit(dataset, dataIdx, classCount, dataIdxsize, fIdx);
		if (split.featureIdx < 0) {
			node.classIndex = std::max_element(classCount.begin(), classCount.end()) - classCount.begin();
			tree.push_back(node);
		}
		else {
			int dataPos = 0;
			int size = dataset.goals.size();
			std::vector<char>& dataIdxLeft = dataIdx;
			std::vector<char> dataIdxRight = dataIdx;
			std::vector<int>& leftClassCount = classCount;
			std::vector<int> rightClassCount = classCount;
			int leftDataSize = dataIdxsize;
			int rightDataSize = dataIdxsize;
			while (dataPos < size) {
				while ((dataPos < size) && (dataIdx[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]] != 1)) {
					++dataPos;
				}
				if (dataset.featuresMatrix[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]][split.featureIdx] < split.splitVal) {
					dataIdxLeft[dataPos] = 1;
					dataIdxRight[dataPos] = 0;
					--rightClassCount[dataset.goals[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]]];
					--rightDataSize;
				}
				else {
					dataIdxLeft[dataPos] = 0;
					dataIdxRight[dataPos] = 1;
					--leftClassCount[dataset.goals[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]]];
					--leftDataSize;
				}
			}
			
			node.featureIndex = split.featureIdx;
			node.threshold = split.splitVal;
			tree.push_back(node);
			int curpos = tree.size() - 1;
			tree[curpos].leftChildIndex = tree.size();
			ConstructTree(dataset, dataIdxLeft, leftClassCount, leftDataSize, fIdx);
			tree[curpos].rightChildIndex = tree.size();
			ConstructTree(dataset, dataIdxRight, rightClassCount, rightDataSize,  fIdx);
		}
	}

	struct TBestSplit {
		int featureIdx;
		double splitVal;
	};

	TBestSplit FindBestSplit(TDataset& dataset, std::vector<char>& dataIdx, std::vector<int>& classCount,int size, std::vector<int>& fIdx) {
		int nf = sqrt(fIdx.size());
#ifdef EACH_NODE_SHUFFLE
		unsigned seed = std::chrono::system_clock::now().time_since_epoch() /
			std::chrono::milliseconds(1);
		shuffle(fIdx.begin(), fIdx.end(), std::default_random_engine(seed));
#endif
		TBestSplit bestSplit;
		bestSplit.featureIdx = -1;
		bestSplit.splitVal = 0;

		double minGini = 1;
		int size = dataset.goals.size();
		int classSize = classCount.size();

		for (int i = 0; i < classSize; ++i) {
			minGini -= (double)(classCount[i] * classCount[i]) / (classSize*classSize);

			std::vector<int> leftClasses(classSize,0);
			std::vector<int> rightClasses;
			for (int fi = 0; fi < nf; ++fi) {
				int featureIdx = fIdx[fi];
				std::vector<double> splitpoints = dataset.splitPointsMatrix[featureIdx];
				int dataPos = 0;
				int leftsize = 0;
				int rightsize = size;
				rightClasses = classCount;
				std::fill(leftClasses.begin(), leftClasses.end(),0);
				while (dataIdx[dataset.sortedByIdxFeaturesMatrix[fi][dataPos]] != 1)
					++dataPos;

				for (double split : splitpoints) {
					while (dataset.featuresMatrix[dataset.sortedByIdxFeaturesMatrix[fi][dataPos]][fi] < split) {
						++leftsize;
						--rightsize;
						++leftClasses[dataset.goals[dataset.sortedByIdxFeaturesMatrix[fi][dataPos]]];
						--rightClasses[dataset.goals[dataset.sortedByIdxFeaturesMatrix[fi][dataPos]]];
						do {
							++dataPos;
						} while (dataIdx[dataset.sortedByIdxFeaturesMatrix[fi][dataPos]] != 1);
					}
					
					double leftGini = 1;
					double rightGini = 1;
					for (int i = 0; i < classSize; ++i) {
						leftGini -= (double)(leftClasses[i] * leftClasses[i]) / (leftsize*leftsize);
						rightGini -= (double)(rightClasses[i] * rightClasses[i]) / (rightsize*rightsize);
					}

					double Gini = (leftGini + rightGini) / 2.;
					if (Gini < minGini) {
						minGini = Gini;
						bestSplit.featureIdx = featureIdx;
						bestSplit.splitVal = split;
					}
				}
		}
		return bestSplit;
	}
	
};