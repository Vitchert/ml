#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include "RForestModel.h"
#include <map>

using namespace std;
struct TInstance {
	vector<double> Features;
	double Goal;
	double Weight;
};
struct TDataset {
	vector<vector<double>> featuresMatrix;
	vector<double> goals;
	vector<double> weights;
	int FeaturesCount;

	map<double,int> classValues;

	vector<vector<size_t>> sortedFeaturesIdxMatrix;

	void sortByFeature() {
		for (int i = 0; i < FeaturesCount; ++i) {
			sortedFeaturesIdxMatrix.push_back(ordered(featuresMatrix, i));
		}
	}

	template <typename T>
	std::vector<size_t> ordered(std::vector<T> const& values, int num) {
		std::vector<size_t> indices(values.size());
		std::iota(begin(indices), end(indices), static_cast<size_t>(0));

		std::sort(
			begin(indices), end(indices),
			[&](size_t a, size_t b) { return values[a][i] < values[b][i]; }
		);
		return indices;
	}

	void ParseFirst(const string& str) {
		stringstream featuresStream(str);
		vector<double> featureset;
		double feature;
		string queryId, url;
		int FeatureCount = 0;

		featuresStream >> queryId; //skip query id
		featuresStream >> feature; //get goal
		goals.push_back(feature);

		classValues.insert({ feature ,1});

		featuresStream >> url; //skip url
		featuresStream >> feature; //get weight
		weights.push_back(1);

		while (featuresStream >> feature) {
			featureset.push_back(feature);
			FeatureCount++;
		}
		FeaturesCount = FeatureCount;
		featuresMatrix.push_back(featureset);
	}

	void Parse(const string& str) {
		stringstream featuresStream(str);
		vector<double> featureset(FeaturesCount);
		double feature;
		string queryId, url;
		int FeatureNumber = 0;

		featuresStream >> queryId; //skip query id
		featuresStream >> feature; //get goal
		goals.push_back(feature);

		map<double, int>::iterator  it = classValues.find(feature);
		if (it != classValues.end())
			++classValues[feature];
		else
			classValues.insert({ feature ,1 });

		featuresStream >> url; //skip url
		featuresStream >> feature; //get weight
		weights.push_back(1);

		while (featuresStream >> feature) {
			featureset[FeatureNumber++] = feature;
		}
		featuresMatrix.push_back(featureset);
	}

	template <typename TSolver>
	TRForestModel Solve(const int& treeCount) {
		TSolver solver;
		solver.AddDataset(*this);
		return solver.Solve(treeCount);
	};

	void ReadFromFile(const string& featuresPath) {
		ifstream featuresIn(featuresPath);

		string featuresString;
		if (getline(featuresIn, featuresString)) 
			ParseFirst(featuresString);
		while (getline(featuresIn, featuresString))
		{
			if (featuresString.empty())
				continue;
			Parse(featuresString);
		}
	}

	enum EIteratorType {
		LearnIterator,
		TestIterator,
	};

	class TCVIterator {
	private:
		const TDataset& ParentDataset;

		TInstance Instance;

		size_t FoldsCount;

		EIteratorType IteratorType;
		size_t TestFoldNumber;

		vector<size_t> InstanceFoldNumbers;
		vector<size_t>::const_iterator Current;

		mt19937 RandomGenerator;
	public:
		TCVIterator(const TDataset& ParentDataset,
			const size_t foldsCount,
			const EIteratorType iteratorType) 
			: ParentDataset(ParentDataset)
			, FoldsCount(foldsCount)
			, IteratorType(iteratorType)
			, InstanceFoldNumbers(ParentDataset.featuresMatrix.size())
		{
		}

		void ResetShuffle() {
			vector<size_t> instanceNumbers(ParentDataset.featuresMatrix.size());
			for (size_t instanceNumber = 0; instanceNumber < ParentDataset.featuresMatrix.size(); ++instanceNumber) {
				instanceNumbers[instanceNumber] = instanceNumber;
			}
			shuffle(instanceNumbers.begin(), instanceNumbers.end(), RandomGenerator);

			for (size_t instancePosition = 0; instancePosition < ParentDataset.featuresMatrix.size(); ++instancePosition) {
				InstanceFoldNumbers[instanceNumbers[instancePosition]] = instancePosition % FoldsCount;
			}
			Current = InstanceFoldNumbers.begin();
		}

		void SetTestFold(const size_t testFoldNumber) {
			TestFoldNumber = testFoldNumber;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}

		bool IsValid() const {
			return Current != InstanceFoldNumbers.end();
		}
		void SetTestMode() {
			IteratorType = TestIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		void SetLearnMode() {
			IteratorType = LearnIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		const TInstance& operator * () {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return Instance;
		}
		const TInstance* operator ->() {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return &Instance;
		}
		TDataset::TCVIterator& operator++() {
			Advance();
			return *this;
		}
	private:
		void Advance() {
			while (IsValid()) {
				++Current;
				if (IsValid() && TakeCurrent()) {
					break;
				}
			}
		}
		bool TakeCurrent() const {
			switch (IteratorType) {
			case LearnIterator: return *Current != TestFoldNumber;
			case TestIterator: return *Current == TestFoldNumber;
			}
			return false;
		}
	};

	class TBaggingIterator {
	private:
		const TDataset& ParentDataset;

		TInstance Instance;

		EIteratorType IteratorType;

		//vector<size_t> InstanceFoldNumbers;
		vector<size_t>::const_iterator Current;

		mt19937 RandomGenerator;
	public:
		vector<size_t> InstanceFoldNumbers;
		TBaggingIterator(const TDataset& ParentDataset,
			const EIteratorType iteratorType)
			: ParentDataset(ParentDataset)
			, IteratorType(iteratorType)
			, InstanceFoldNumbers(ParentDataset.featuresMatrix.size())
		{
		}

		void ResetShuffle(size_t seed) {
			RandomGenerator.seed(seed);
			int size = ParentDataset.featuresMatrix.size();
			vector<size_t> instanceNumbers(size,0); //TEST = 0
	
			for (size_t instanceNumber = 0; instanceNumber < size; ++instanceNumber) {
				instanceNumbers[RandomGenerator() % size] = 1; //LEARN = 1
			}
			InstanceFoldNumbers = instanceNumbers;
			Current = InstanceFoldNumbers.begin();
		}

		bool IsValid() const {
			return Current != InstanceFoldNumbers.end();
		}
		void SetTestMode() {
			IteratorType = TestIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		void SetLearnMode() {
			IteratorType = LearnIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		const TInstance& operator * () {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return Instance;
		}
		const TInstance* operator ->() {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return &Instance;
		}
		TDataset::TBaggingIterator& operator++() {
			Advance();
			return *this;
		}
	private:
		void Advance() {
			while (IsValid()) {
				++Current;
				if (IsValid() && TakeCurrent()) {
					break;
				}
			}
		}
		bool TakeCurrent() const {
			switch (IteratorType) {
			case LearnIterator: return *Current == 1;
			case TestIterator: return *Current == 0;
			}
			return false;
		}
	};

	TCVIterator CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType = LearnIterator) const {
		return TCVIterator(*this, foldsCount, iteratorType);
	}
	TBaggingIterator BaggingIterator( const EIteratorType iteratorType = LearnIterator) const {
		return TBaggingIterator(*this, iteratorType);
	}
	/*
	template <typename TSolver>
	TRForestModel SolveCrossValidation(TCVIterator& iterator) {
		TSolver solver;
		TInstance instance;

		while (iterator.IsValid()) {
			instance = *iterator;
			solver.Add(instance.Features, instance.Goal, instance.Weight);
			++iterator;
		}

		return solver.Solve();
	};
	*/
};