package main;

import java.io.IOException;
import java.util.ArrayList;

import data_structure.DenseMatrix;
import data_structure.Rating;
import data_structure.SparseMatrix;
import utils.Printer;
import algorithms.MF_fastALS;
import algorithms.MFbpr;
import algorithms.MF_Pair1AALS;
import algorithms.MF_Pair1AALS_Sampled;
import algorithms.MF_Pair2AALS;
import algorithms.MF_Point1AALS;
import algorithms.MF_Point2AALS;
import algorithms.MF_RankingALS;
import algorithms.ItemPopularity;

public class main_MF extends main {
	public static void main(String argv[]) throws IOException {
		String dataset_name = "buy";
		String sidedataset_name = "ipv";
		String sidedataset2_name = "collect";
		String method = "FastALS";
		double w0 = 10;
		double w1 = 1;
		double w2 = 1;
		double r1 = 1;
		double r2 = 1;
		boolean showProgress = false;
		boolean showLoss = true;
		int factors = 64;
		int maxIter = 500;
		double reg = 0.01;
		double alpha = 0.75;
		double beta = 0.2;
		double ratio = 0;
		double gamma1 = 0;
		double gamma2 = 0;
		double gamma3 = 0;
		//whether to randomly select 1 record per user from training set as validation set
		boolean validation = false; 
		boolean eval_topk = false;
		int iter_adaptive = 0;
		
		if (argv.length > 0) {
			dataset_name = argv[0];
			method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8) alpha = Double.parseDouble(argv[8]);
			if (method.equalsIgnoreCase("RankingALS")) {
				sidedataset_name = argv[9];
				r1 = Double.parseDouble(argv[10]);
				sidedataset2_name = argv[11];
				r2 = Double.parseDouble(argv[12]);
			}else {
				if (argv.length > 9) {
					sidedataset_name = argv[9];
					w1 = Double.parseDouble(argv[10]);
				}
				if (argv.length > 11) {
					beta = Double.parseDouble(argv[11]);
				}
				if (argv.length > 12) {
					r1 = Double.parseDouble(argv[12]);
					gamma1 = Double.parseDouble(argv[12]);
				}
				if (method.equalsIgnoreCase("Point2AALS")) {
					sidedataset2_name = argv[13];
					w2 = Double.parseDouble(argv[14]);
					r2 = Double.parseDouble(argv[15]);
					eval_topk = Boolean.parseBoolean(argv[16]);
					validation = Boolean.parseBoolean(argv[17]);
				}else {
					if (argv.length > 13){
						w2 = Double.parseDouble(argv[13]);
						gamma2 = Double.parseDouble(argv[13]);
					}
					if (method.equalsIgnoreCase("Pair2AALS")) {
						sidedataset2_name = argv[14];
						w2 = Double.parseDouble(argv[15]);
						gamma3 = Double.parseDouble(argv[16]);
						eval_topk = Boolean.parseBoolean(argv[17]);
						validation = Boolean.parseBoolean(argv[18]);
					}else {
						if (argv.length > 14){
							validation = Boolean.parseBoolean(argv[14]);
						}
						if (argv.length > 15){
							iter_adaptive = Integer.parseInt(argv[15]);
						}
					}
				}
			}
		}
		//ReadRatings_GlobalSplit("data/" + dataset_name + ".rating", 0.1);
		if (!dataset_name.contains("tmall"))
			ReadRatings_HoldOneOut(dataset_name);
		else
			ReadRatings_HoldOneOut_Tmall(dataset_name);
		if (method.contains("AALS") || method.equalsIgnoreCase("RankingALS")) {
			ReadSideRatings(sidedataset_name, r1);
		}
		if (method.contains("2AALS") || method.equalsIgnoreCase("RankingALS")) {
			ReadSideRatings2(sidedataset2_name, r2);
		}
		if (method.equalsIgnoreCase("SplitData")) {
			SplitDataFile_HoldOneOut_Tmall(dataset_name,dataset_name+"-train",dataset_name+"-test");
			return;
		}
		System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, alpha=%.2f, w1=%.6f, r1=%.2f, w2=%.6f, beta=%.2f\n",
				method, showProgress, factors, maxIter, reg, w0, alpha, w1, r1, w2, beta);
		System.out.println("====================================================");
		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings, topK, threadNum);
		evaluate_model(popularity, "Popularity");
		
		double init_mean = 0;
		double init_stdev = 0.01;
		
		if (method.equalsIgnoreCase("fastals")) {
			MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(fals, "MF_fastALS");
		}
		
		ratio = beta;
		
		if (method.equalsIgnoreCase("RankingALS")) {
			MF_RankingALS eALSplusView = new MF_RankingALS(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss, trainSideMatrix2, trainSideMatrix, eval_topk);
			evaluate_model(eALSplusView, "MF_RankingALS");
		}
		
		if (method.equalsIgnoreCase("Pair1AALS")) {
			MF_Pair1AALS eALSplusView = new MF_Pair1AALS(trainMatrix, testRatings, trainSideMatrix, topK, threadNum,
					factors, maxIter, w0, w1, alpha, reg, beta, gamma1, gamma2, init_mean, init_stdev, showProgress, showLoss, validation);
			evaluate_model(eALSplusView, "MF_Pair1AALS");
		}
		if (method.equalsIgnoreCase("Pair2AALS")) {
			MF_Pair2AALS eALSplusView = new MF_Pair2AALS(trainMatrix, testRatings, trainSideMatrix, topK, threadNum,
					factors, maxIter, w0, w1, alpha, reg, beta, gamma1, gamma2, init_mean, init_stdev, showProgress, showLoss, trainSideMatrix2, w2, gamma3, eval_topk);
			evaluate_model(eALSplusView, "MF_Pair2AALS");
		}
		
		if (method.equalsIgnoreCase("Pair1AALS-Sampled")) {
			MF_Pair1AALS_Sampled eALSplusView = new MF_Pair1AALS_Sampled(trainMatrix, testRatings, trainSideMatrix, topK, threadNum,
					factors, maxIter, w0, w1, alpha, reg, beta, gamma1, gamma2, init_mean, init_stdev, showProgress, showLoss, validation, iter_adaptive);
			evaluate_model(eALSplusView, "MF_Pair1AALS_Sampled");
		}
		
		if (method.equalsIgnoreCase("Point1AALS")) {
			MF_Point1AALS fals = new MF_Point1AALS(trainMatrix, testRatings, trainSideMatrix, topK, threadNum,
					factors, maxIter, w0, w1, alpha, reg, init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(fals, "MF_Point1AALS");
		}
		if (method.equalsIgnoreCase("Point2AALS")) {
			MF_Point2AALS fals = new MF_Point2AALS(trainMatrix, testRatings, trainSideMatrix, topK, threadNum,
					factors, maxIter, w0, w1, alpha, reg, init_mean, init_stdev, showProgress, showLoss, trainSideMatrix2, w2, eval_topk);
			evaluate_model(fals, "MF_Point2AALS");
		}
		
		if (method.equalsIgnoreCase("bpr")) {
			MFbpr bpr = new MFbpr(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, false, reg, init_mean, init_stdev, showProgress);
			evaluate_model(bpr, "MFbpr");
		}
		
	} // end main
}
