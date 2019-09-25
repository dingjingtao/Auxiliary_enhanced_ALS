package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;
import data_structure.SparseVector;
import happy.coding.math.Randoms;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

import utils.Printer;

/**
 * Fast ALS for weighted matrix factorization (with imputation)
 * 
 * @author xiangnanhe
 */
public class MF_RankingALS extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; // number of latent factors.
	int maxIter = 500; // maximum iterations.
	double reg = 0.01; // regularization parameters
	double w0 = 1;
	double init_mean = 0; // Gaussian mean for init V
	double init_stdev = 0.01; // Gaussian std-dev for init V

	/** Model parameters to learn */
	public DenseMatrix U; // latent vectors for users
	public DenseMatrix V; // latent vectors for items

	/** Caches */
	DenseMatrix SU;
	DenseMatrix SV;
	double[] prediction_users, prediction_items;
	double[] rating_users, rating_items;
	double[] w_users, w_items;

	boolean showProgress;
	boolean showLoss;
	boolean eval_topk;

	// weight for each positive instance in trainMatrix
	SparseMatrix W;

	// weight for negative instances on item i.
	double[] Wi;

	// weight of new instance in online learning
	public double w_new = 1;
	
	public double sumSupport;
	public DenseVector supportVector;

	public MF_RankingALS(SparseMatrix buy, ArrayList<Rating> testRatings, int topK, int threadNum, int factors,
			int maxIter, double w0, double alpha, double reg, double init_mean, double init_stdev, boolean showProgress,
			boolean showLoss, SparseMatrix trainSideMatrix2, SparseMatrix trainSideMatrix1, boolean eval_topk) {
		super(buy, testRatings, topK, threadNum);
		userCount = buy.length()[0];
		itemCount = buy.length()[1];
		this.trainMatrix = new SparseMatrix(userCount, itemCount);
		for (int  u = 0;u<userCount;u++)
			for (int i = 0;i <itemCount;i++) {
				if (buy.getValue(u, i) != 0)
					trainMatrix.setValue(u, i, 1);
				else if (trainSideMatrix1.getValue(u, i) != 0)
					trainMatrix.setValue(u, i,trainSideMatrix1.getValue(u, i));
				else if (trainSideMatrix2.getValue(u, i) != 0)
					trainMatrix.setValue(u, i,trainSideMatrix2.getValue(u, i));
			}		
		System.out.printf("whole train is %d size\n", trainMatrix.nonZeroCount());
		this.factors = factors;
		this.maxIter = maxIter;
		this.w0 = w0;
		this.reg = reg;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showLoss = showLoss;
		this.showProgress = showProgress;
		this.eval_topk = eval_topk;
		
		// Init caches
		prediction_users = new double[userCount];
		prediction_items = new double[itemCount];
		rating_users = new double[userCount];
		rating_items = new double[itemCount];
		w_users = new double[userCount];
		w_items = new double[itemCount];

		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
		//initS();
		
		init_rankals();
	}

	public void setTrain(SparseMatrix trainMatrix) {
		this.trainMatrix = new SparseMatrix(trainMatrix);
		W = new SparseMatrix(userCount, itemCount);
		for (int u = 0; u < userCount; u++)
			for (int i : this.trainMatrix.getRowRef(u).indexList())
				W.setValue(u, i, 1);
	}

	private void init_rankals() {
		supportVector = new DenseVector(itemCount);
		sumSupport = 0;
		for (int i =0;i<itemCount;i++) {
			double s = trainMatrix.getColRef(i).indexList().size();
			supportVector.set(i, s); 
			sumSupport += s;
		}
	}
	// Init SU and SV
	private void initS() {
		SU = U.transpose().mult(U);
		// Init SV as V^T Wi V
		SV = new DenseMatrix(factors, factors);
		for (int f = 0; f < factors; f++) {
			for (int k = 0; k <= f; k++) {
				double val = 0;
				for (int i = 0; i < itemCount; i++)
					val += V.get(i, f) * V.get(i, k) * Wi[i];
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}
	}

	// remove
	public void setUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
		initS();
	}

	public void buildModel() throws IOException {
		// System.out.println("Run for FastALS. ");
		double loss_pre = Double.MAX_VALUE;
		
		if (true) {
			for (int iter = 0; iter < maxIter; iter++) {
				Long start = System.currentTimeMillis();

				// Update user latent vectors
				update_useranditem();
				//reshape_UV();
				if ((iter >= maxIter - 10) || (iter % 1 == 0)) {
					showProgress(iter, start, testRatings);
					if (eval_topk)
						evaluate_topk(testRatings);
			}
				
				// Show loss
				if (showLoss)
					loss_pre = showLoss(iter, start, loss_pre);
			}
		}
	}


	// Run model for one iteration
	public void runOneIteration() {
		// Update user latent vectors
		for (int u = 0; u < userCount; u++) {
			update_user(u);
		}

		// Update item latent vectors
		for (int i = 0; i < itemCount; i++) {
			update_item(i);
		}
	}

	protected void update_user(int u ) {
		
	}
	
	// see what happened
	private void reshape_UV() {
		double Umax = 0;
		double Vmax = 0;
		for (int i = 0;i<itemCount;i++) 
			for (int k =0;k<factors;k++)
				if (Vmax < Math.abs(V.get(i, k)))
						Vmax = Math.abs(V.get(i, k));		
		for (int u =0;u<userCount;u++) 
			for (int k =0;k<factors;k++)
				if (Umax < Math.abs(U.get(u, k)))
						Umax = Math.abs(U.get(u, k));
		U = U.scale(1/Umax);
		V = V.scale(1/Vmax);
	}
	
	protected void update_useranditem() {
		DenseVector sum_sq = new DenseVector(factors); 
		DenseMatrix sum_sqq = new DenseMatrix(factors,factors);
		for (int i = 0;i<itemCount;i++) {
			DenseVector qi = V.row(i);
			double sj = supportVector.get(i);
			sum_sq = sum_sq.add(qi.scale(sj));
			sum_sqq = sum_sqq.add(qi.outer(qi).scale(sj));
		}
		for (int u = 0;u<userCount;u++) {
            ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
			if (itemList.size( ) == 0) {
	    		continue;
	    	}
			DenseMatrix sum_cqq = new DenseMatrix(factors, factors);
            DenseVector sum_cq = new  DenseVector(factors);
            DenseVector sum_cqr = new  DenseVector(factors);
            DenseVector sum_sqr = new  DenseVector(factors);

            DenseVector Ru = new  DenseVector(itemCount);
            double sum_c = 0;
            double sum_sr = 0, sum_cr = 0;
            for (int i = 0;i<itemCount;i++) {
            	Ru.set(i, trainMatrix.getValue(u, i));
            	if (trainMatrix.getValue(u, i)!=0)
            			sum_c++;         	
            }

    		for (int i:itemList) {
    			DenseVector qi = V.row(i);
    			double rui = trainMatrix.getValue(u,i);
                sum_cqq = sum_cqq.add(qi.outer(qi));
                sum_cq = sum_cq.add(qi);
                sum_cqr = sum_cqr.add(qi.scale(rui));
                double si = supportVector.get(i);
                sum_sr += si * rui;
                sum_cr += rui;
                sum_sqr = sum_sqr.add(qi.scale(si * rui));
    		}
            DenseMatrix M = sum_cqq.scale(sumSupport).minus(sum_cq.outer(sum_sq)).minus(sum_sq.outer(sum_cq))
                    .add(sum_sqq.scale(sum_c));
            DenseVector y = sum_cqr.scale(sumSupport).minus(sum_cq.scale(sum_sr)).minus(sum_sq.scale(sum_cr))
                    .add(sum_sqr.scale(sum_c));

            DenseVector pu = M.inv().mult(y);
//            if (u <= 1) {
//            	System.out.printf("user:%d, U:%f,%f\n",u,U.get(u, 0),U.get(u, 1));
//            	System.out.printf("user:%d, U:%f,%f\n",u,pu.get(0),pu.get(1));
//            }
            for (int k =0;k<factors;k++)
            	U.set(u, k, pu.get(k));      
		}
		
		 Map<Integer, Double> m_sum_sr = new HashMap<>();
         Map<Integer, Double> m_sum_cr = new HashMap<>();
         Map<Integer, Double> m_sum_c = new HashMap<>();
         Map<Integer, DenseVector> m_sum_cq = new HashMap<>();
         for (int u =0;u<userCount;u++) {
            ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
 			if (itemList.size( ) == 0) {
 	    		continue;
 	    	}
             double sum_sr = 0, sum_cr = 0, sum_c = itemList.size();
             DenseVector sum_cq = new DenseVector(factors);

             for (int i :itemList) {
                 double ruj = trainMatrix.getValue(u,i);
                 double sj = supportVector.get(i);

                 sum_sr += sj * ruj;
                 sum_cr += ruj;
                 sum_cq = sum_cq.add(V.row(i));
             }
             m_sum_sr.put(u, sum_sr);
             m_sum_cr.put(u, sum_cr);
             m_sum_c.put(u, sum_c);
             m_sum_cq.put(u, sum_cq);
         }
         for (int i = 0; i < itemCount; i++) {
             // for each item
      	   int numFactors = factors;
             DenseMatrix sum_cpp = new DenseMatrix(numFactors, numFactors);
             DenseMatrix sum_p_p_c = new DenseMatrix(numFactors, numFactors);
             DenseVector sum_p_p_cq = new DenseVector(numFactors);
             DenseVector sum_cpr = new DenseVector(numFactors);
             DenseVector sum_c_sr_p = new DenseVector(numFactors);
             DenseVector sum_cr_p = new DenseVector(numFactors);
             DenseVector sum_p_r_c = new DenseVector(numFactors);

             double si =  supportVector.get(i);
             ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
             for (int u =0;u<userCount;u++) {
                 ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
     			if (itemList.size( ) == 0) {
     	    		continue;
     	    	}
                 DenseVector pu = U.row(u);
                 double rui = trainMatrix.getValue(u,i);

                 DenseMatrix pp = pu.outer(pu);
                 sum_cpp = sum_cpp.add(pp);
                 sum_p_p_cq = sum_p_p_cq.add(pp.mult(m_sum_cq.get(u)));
                 sum_p_p_c = sum_p_p_c.add(pp.scale(m_sum_c.get(u)));
                 sum_cr_p = sum_cr_p.add(pu.scale(m_sum_cr.get(u)));

                 if (rui > 0) {
                     sum_cpr = sum_cpr.add(pu.scale(rui));
                     sum_c_sr_p = sum_c_sr_p.add(pu.scale(m_sum_sr.get(u)));
                     sum_p_r_c = sum_p_r_c.add(pu.scale(rui * m_sum_c.get(u)));
                 }
             }
             DenseMatrix subtract = sum_cpp.add(si + 1);
             DenseMatrix M = sum_cpp.scale(sumSupport).add(sum_p_p_c.scale(si)).minus(subtract);
             DenseVector y = sum_cpp.mult(sum_sq).add(sum_cpr.scale(sumSupport)).minus(sum_c_sr_p)
                     .add(sum_p_p_cq.scale(si)).minus(sum_cr_p.scale(si)).add(sum_p_r_c.scale(si));
             DenseVector qi = M.inv().mult(y.minus(subtract.mult(V.row(i))));
//             if (i <= 1) {
//             	System.out.printf("item:%d, U:%f,%f\n",i,V.get(i, 0),V.get(i, 1));
//             	System.out.printf("item:%d, U:%f,%f\n",i,qi.get(0),qi.get(1));
//             }
             for (int k =0;k<factors;k++)
             	V.set(i, k, qi.get(k)); 
         }	
	}
	
	protected void update_item() {
		  
	}

	protected void update_item(int i) {
		ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
		if (userList.size() == 0)
			return; // item has no ratings.
		// prediction cache for the item
		for (int u : userList) {
			prediction_users[u] = predict(u, i);
			rating_users[u] = trainMatrix.getValue(u, i);
			w_users[u] = W.getValue(u, i);
		}

		DenseVector oldVector = V.row(i);
		for (int f = 0; f < factors; f++) {
			// O(K) complexity for the w0 part
			double numer = 0, denom = 0;
			for (int k = 0; k < factors; k++) {
				if (k != f)
					numer -= V.get(i, k) * SU.get(f, k);
			}
			numer *= Wi[i];

			// O(Ni) complexity for the positive ratings part
			for (int u : userList) {
				prediction_users[u] -= U.get(u, f) * V.get(i, f);
				numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * U.get(u, f);
				denom += (w_users[u] - Wi[i]) * U.get(u, f) * U.get(u, f);
			}
			denom += Wi[i] * SU.get(f, f) + reg;

			// Parameter update
			V.set(i, f, numer / denom);
			// Update the prediction cache for the item
			for (int u : userList)
				prediction_users[u] += U.get(u, f) * V.get(i, f);
		} // end for f

		// Update the SV cache
		for (int f = 0; f < factors; f++) {
			for (int k = 0; k <= f; k++) {
				double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k) * Wi[i]
						+ V.get(i, f) * V.get(i, k) * Wi[i];
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}
	}

	public double showLoss(int iter, long start, double loss_pre) {
		long start1 = System.currentTimeMillis();
		double loss_cur = loss();
		String symbol = loss_pre >= loss_cur ? "-" : "+";
		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f [%s]\n", iter, Printer.printTime(start1 - start), symbol,
				loss_cur, Printer.printTime(System.currentTimeMillis() - start1));
		return loss_cur;
	}

	// Fast way to calculate the loss function
	public double loss() {
		double L = 0;
		double []r = new double [itemCount];
		for (int u =0;u<userCount;u++) {
			double lu = 0;
			for (int i =0;i<itemCount;i++)
				r[i] = predict(u,i);
			ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
			int  sumi = itemList.size();
			double sumri = 0,sumrj=0,sumssi=0,sumssj = 0;
			for (int i = 0;i<itemCount;i++) {
				if (trainMatrix.getValue(u, i)!=0) {
					sumri += r[i];
					sumssi += r[i] * r[i];					
				}
				else {
					sumrj += r[i];
					sumssj += r[i];
				}					
			}
			lu = (itemCount - sumi)*sumssi + sumi*sumssj - 2*sumri*sumrj - 2*((itemCount - sumi)*sumri - sumi*sumrj);		
			L += lu;
		}
		return L;
	}

	@Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}

	@Override
	public void updateModel(int u, int i) {
		trainMatrix.setValue(u, i, 1);
		W.setValue(u, i, w_new);
		if (Wi[i] == 0) { // an new item
			Wi[i] = w0 / itemCount;
			// Update the SV cache
			for (int f = 0; f < factors; f++) {
				for (int k = 0; k <= f; k++) {
					double val = SV.get(f, k) + V.get(i, f) * V.get(i, k) * Wi[i];
					SV.set(f, k, val);
					SV.set(k, f, val);
				}
			}
		}

		for (int iter = 0; iter < maxIterOnline; iter++) {
			update_user(u);

			update_item(i);
		}
	}

	/*
	 * // Raw way to calculate the loss function public double loss() { double L =
	 * reg * (U.squaredSum() + V.squaredSum()); for (int u = 0; u < userCount; u ++)
	 * { double l = 0; for (int i : trainMatrix.getRowRef(u).indexList()) { l +=
	 * Math.pow(trainMatrix.getValue(u, i) - predict(u, i), 2); } l *= (1 - w0); for
	 * (int i = 0; i < itemCount; i ++) { l += w0 * Math.pow(predict(u, i), 2); } L
	 * += l; } return L; }
	 */
}
