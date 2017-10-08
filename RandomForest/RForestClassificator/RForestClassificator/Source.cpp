#include <iostream>
#include "RForestSolver.h"

#define EACH_NODE_SHUFFLE
int main(int argc, const char** argv) {
	(void)argc;
	

	std::string featuresFilePath = argv[1];
	std::string trees = argv[2];
	int treecount = stoi(trees);
	//string modelFilePath = argv[3];
	TDataset dataset;
	dataset.ReadFromFile(featuresFilePath);

	//TRForestSolver solver;
	//solver.AddDataset(dataset);
	//TRForestModel model = solver.Solve(treecount);

	int rounds = 1;
	int folds = 3;
	TInstance instance;
	TDataset::TCVIterator iterator = dataset.CrossValidationIterator(folds);		
	double SSEmean = 0;
	int prediction;
	for (int i = 0; i < rounds; ++i) {
		cout << "CV round " <<i<<endl;
		
		iterator.ResetShuffle();
		double SSE = 0;
		int testsize = 1;
		for (int j = 0; j < folds; ++j) {
			iterator.SetLearnMode();
			iterator.SetTestFold(j);
			TRForestSolver solver;
			while (iterator.IsValid()) {
				instance = *iterator;
				solver.Add(instance.Features, instance.Goal, instance.Weight);
				++iterator;
			}

			TRForestModel model = solver.Solve(treecount);
			iterator.SetTestMode();
	
			double summ = 0;
			int predictionsize = 0;
			while (iterator.IsValid()) {
				instance = *iterator;
				prediction = model.Prediction(instance.Features);
				if (prediction != instance.Goal)
					summ++;
				++iterator;
				++predictionsize;
			}
			if (predictionsize) 
				summ /= predictionsize;
			if (summ != 0) {
				SSE += (summ - SSE) / (testsize++);
			}
			else {
				cout <<"j "<< j << " s " << summ << endl;
				testsize++;
			}
		}
		cout << "SSE " << SSE << endl;
		cout << SSE << endl;
		SSEmean += (SSE - SSEmean) / (i + 1);
	}
	cout << endl << "Mean "<<SSEmean << endl;


	//if (mode == "learn") {
	//	
	//	string learningMode = argv[4];
	//
	//	TLinearModel linearModel;
	//
	//	if (learningMode == "welford_lr") {
	//		linearModel = dataset.Solve<TWelfordSolver>();
	//	}
	//	else {
	//		cout << "Unknown learning mode.\n";
	//	}
	//
	//	linearModel.SaveToFile(modelFilePath);
	//}
	//
	//if (mode == "predict") {
	//	string predictionsPath = argv[4];
	//
	//	ofstream predictionsOut(predictionsPath);
	//	predictionsOut.precision(20);
	//
	//	TLinearModel linearModel = TLinearModel::LoadFromFile(modelFilePath);
	//
	//	for (vector<vector<double>>::iterator it = dataset.featuresMatrix.begin(); it != dataset.featuresMatrix.end(); ++it) {
	//		predictionsOut << linearModel.Prediction(*it) << "\n";
	//	}
	//}
	//
	//if (mode == "predictError") {
	//	string predictionsPath = argv[4];
	//
	//	TLinearModel linearModel = TLinearModel::LoadFromFile(modelFilePath);
	//
	//	linearModel.PredictionError(dataset.featuresMatrix, dataset.goals, dataset.weights, predictionsPath);
	//}

	//if (mode == "crossValidation") {
	//	string learningMode = argv[4];
	//	string SSEPath = argv[5];
	//	int rounds = stoi(argv[6]);
	//	int folds = stoi(argv[7]);
	//
	//	ofstream SSEOut(SSEPath);
	//	SSEOut.precision(20);
	//
	//	TLinearModel linearModel;
	//	TInstance instance;
	//	TDataset::TCVIterator iterator = dataset.CrossValidationIterator(folds);		
	//	TKahan SSEmean = 0;
	//
	//	if (learningMode == "welford_lr") {
	//		double prediction;
	//		for (int i = 0; i < rounds; ++i) {
	//			cout << "CV round " <<i<<endl;
	//
	//			iterator.ResetShuffle();
	//			TKahan SSE = 0;
	//			int testsize = 1;
	//			for (int j = 0; j < folds; ++j) {
	//				iterator.SetLearnMode();
	//				iterator.SetTestFold(j);
	//				linearModel = dataset.SolveCrossValidation<TWelfordSolver>(iterator);
	//
	//				iterator.SetTestMode();
	//
	//				TKahan summ = 0;
	//				int predictionsize = 1;
	//				while (iterator.IsValid()) {
	//					instance = *iterator;
	//					prediction = linearModel.Prediction(instance.Features);
	//					summ += (instance.Weight * (prediction - instance.Goal) * (prediction - instance.Goal) - summ)/(predictionsize++);
	//					++iterator;
	//				}
	//				if (summ != 0) {
	//					SSE += (summ - SSE) / (testsize++);
	//				}
	//				else {
	//					SSEOut <<"j "<< j << " s " << summ << endl;
	//				}
	//			}
	//			cout << "SSE " << SSE << endl;
	//			SSEOut << SSE << endl;
	//			SSEmean += (SSE - SSEmean) / (i + 1);
	//		}
	//		SSEOut << endl << "Mean "<<SSEmean << endl;
	//	}
	//	else {
	//		cout << "Unknown learning mode.\n";
	//	}
	//
	//}
	return 0;
}

