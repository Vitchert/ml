#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

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

	void ParseFirst(const string& str) {
		stringstream featuresStream(str);
		vector<double> featureset;
		double feature;
		string queryId, url;
		int FeatureCount = 0;

		featuresStream >> queryId; //skip query id
		featuresStream >> feature; //get goal
		goals.push_back(feature);

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

		featuresStream >> url; //skip url
		featuresStream >> feature; //get weight
		weights.push_back(1);

		while (featuresStream >> feature) {
			featureset[FeatureNumber++] = feature;
		}
		featuresMatrix.push_back(featureset);
	}

	template <typename TSolver>
	TLinearModel Solve() {
		TSolver solver;
		solver.AddDataset(*this);
		return solver.Solve();
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

	TDataset InjureData(const double injureFactor, const double injureOffset) {
		TDataset injuredDataset(*this);

		for (vector<double>& featureSet : featuresMatrix) {
			for (double& feature : featureSet) {
			feature = feature * injureFactor + injureOffset;
			}
		}
		for (double& goal : goals) {
			goal = goal *  injureFactor + injureOffset;
		}

		return injuredDataset;
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

	TCVIterator CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType = LearnIterator) const {
		return TCVIterator(*this, foldsCount, iteratorType);
	}

	template <typename TSolver>
	TLinearModel SolveCrossValidation(TCVIterator& iterator) {
		TSolver solver;
		TInstance instance;

		while (iterator.IsValid()) {
			instance = *iterator;
			solver.Add(instance.Features, instance.Goal, instance.Weight);
			++iterator;
		}

		return solver.Solve();
	};
};