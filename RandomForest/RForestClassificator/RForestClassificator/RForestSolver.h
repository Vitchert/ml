#pragma once
#include <iostream>
#include "RForestModel.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono> 
#include "windows.h"

class TRForestSolver {
private:
	TDataset dataset;
public:
	void AddDataset(const TDataset& data) {
		dataset = data;
	}
	void Add(const vector<double>& features, const double goal, const double weight = 1.) {
		dataset.featuresMatrix.push_back(features);
		dataset.goals.push_back(goal);
		dataset.weights.push_back(weight);
	}

	std::mutex g_lock;
	std::mutex sem;
	int semCount = 6;

	void threadFunction(TRForestModel& model, int num)
	{
		std::cout << "Tree n" << num << endl;
		TDecisionTree dTree;
		std::vector<char> dataIdx(dataset.goals.size(),0);
		std::vector<char> testIdx(dataset.goals.size(), 0);
		TDataset::TBaggingIterator it = dataset.BaggingIterator();
		int datasize = 0;
		it.ResetShuffle(num);
		it.SetLearnMode();
		int size = it.InstanceFoldNumbers.size();
		for (int i = 0; i < size; ++i) {
			if (it.InstanceFoldNumbers[i]) {
				dataIdx[i] = 1;
				++datasize;
			}
			else
				testIdx[i]=1;
		}

		int featureCount = dataset.featuresMatrix[0].size();
		std::vector<int> fIdx(featureCount, 0);
		for (int i = 0; i < featureCount; ++i) {
			fIdx[i] = i;
		}

		unsigned seed = std::chrono::system_clock::now().time_since_epoch() /
			std::chrono::milliseconds(1);
		shuffle(fIdx.begin(), fIdx.end(), std::default_random_engine(seed+num));
		//fIdx.resize((fIdx.size() * 2) / 3);
		dTree.ConstructTree(dataset, dataIdx, dataset.classCount, datasize, fIdx);
		g_lock.lock();
		model.forest.push_back(dTree);
		g_lock.unlock();
		//TODO OOB
		/*
		int cl;
		int wrong = 0;
		for (size_t idx : testIdx) {
			cl = dTree.Prediction<double>(dataset.featuresMatrix[idx]);
			if (cl != dataset.goals[idx])
				++wrong;
		}
		std::cout << "OOB tree n" << num << " " << (double)wrong / dataIdx.size() << endl;
		*/
		sem.lock();
		++semCount;
		sem.unlock();
	}

	TRForestModel Solve(int treeCount) {
		dataset.SortFeatures();
		dataset.CalculateSplitpoints();
		dataset.PrepareGoals();

		TRForestModel model;

		for (int i = 0; i < treeCount; ++i) {
			while (true) {
				sem.lock();
				if (semCount > 0) {
					--semCount;
					thread(&TRForestSolver::threadFunction, this, std::ref(model), i).detach();
					sem.unlock();
					break;
				}
				else {
					sem.unlock();
					Sleep(500);
				}				
			}			
		}
		while (true) {
			cout << "sleep\n";
			Sleep(3000);
			g_lock.lock();
			if (model.forest.size() == treeCount)
				break;
			g_lock.unlock();
		}
		g_lock.unlock();
		return model;
	}
	
};