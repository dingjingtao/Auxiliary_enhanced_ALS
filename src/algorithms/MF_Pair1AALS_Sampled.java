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

import utils.Printer;

/**
 * Pairwise Auxiliary-enhanced eALS for weighted matrix factorization (Sample-based Solution)
 * 
 * @author jingtaoding
 */
public class MF_Pair1AALS_Sampled extends TopKRecommender {
	/** Model priors to set. */
	int factors = 10; // number of latent factors.
	int maxIter = 500; // maximum iterations.
	double reg = 0.01; // regularization parameters
	double lr = 0.01;
	double init_mean = 0; // Gaussian mean for init V
	double init_stdev = 0.01; // Gaussian std-dev for init V
	double gamma1 = 0;
	double gamma2 = 0;
	/** The number of users. */
	public int userCount_side;
	/** The number of items. */
	public int itemCount_side;

	public SparseMatrix trainSideMatrix;

	/** Model parameters to learn */
	public DenseMatrix U; // latent vectors for users
	public DenseMatrix V; // latent vectors for items

	public Integer [][]viewdata;
	public Integer [][]buydata;


	boolean showProgress;
	boolean showLoss;
	boolean validation;
	public ArrayList<Rating> validationRatings;
	
	boolean adaptive,adaptive_flag;
	int iter_adaptive;
	
	Random rand = new Random();

	public MF_Pair1AALS_Sampled(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, SparseMatrix trainSideMatrix,
			int topK, int threadNum, int factors, int maxIter, double lr, double w1, double alpha, double reg,
			double beta, double gamma1, double gamma2, double init_mean, double init_stdev, boolean showProgress,
			boolean showLoss, boolean validation, int iter_adaptive) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.trainSideMatrix = new SparseMatrix(trainSideMatrix);
		this.userCount_side = trainSideMatrix.length()[0];
		this.itemCount_side = trainSideMatrix.length()[1];
		this.factors = factors;
		this.maxIter = maxIter;
		this.lr = lr;
		this.reg = reg;
		this.gamma1 = gamma1;
		this.gamma2 = gamma2;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showLoss = showLoss;
		this.showProgress = showProgress;
		this.validation = validation;
		this.validationRatings = new ArrayList<Rating>();
		this.adaptive = showProgress;
		this.adaptive_flag = false;
		this.iter_adaptive = iter_adaptive;

		// Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);
	}

	public void buildModel() throws IOException {
		double loss_pre = Double.MAX_VALUE;
		double hr_prev = 0.0;
		int nonzeros = trainMatrix.itemCount();
		buydata = new Integer[userCount][];
		viewdata = new Integer[userCount][];
		for (int i =0;i<userCount;i++) {
			ArrayList<Integer> itemList = trainMatrix.getRowRef(i).indexList();
			ArrayList<Integer> viewList = trainSideMatrix.getRowRef(i).indexList();
			buydata[i] = itemList.toArray(new Integer [itemList.size()]);
			viewdata[i] = viewList.toArray(new Integer [viewList.size()]);
		}
		
//		Random rand = new Random();
//		SparseMatrix trainMatrixAll = new SparseMatrix(trainMatrix);
		for (int iter = 0; iter < maxIter; iter++) {
			Long start = System.currentTimeMillis();
			if (validation) {
				if (!validationRatings.isEmpty())
					for(Rating rating:validationRatings) {
						trainMatrix.setValue(rating.userId, rating.itemId, 1.0);
					}
//					trainMatrix = new SparseMatrix(trainMatrixAll);
				validationRatings.clear();
				for (int u = 0; u < userCount; u++) {
					ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
					if (itemList.size()<=0)
						continue;
					int r = rand.nextInt(itemList.size());
					trainMatrix.setValue(u, itemList.get(r), 0.0);
					Rating rating = new Rating(u, itemList.get(r).intValue(), (float)1.0, (long)0);
					validationRatings.add(rating);
				}
			}
			
			// Each training epoch
			for (int s = 0; s < nonzeros; s ++) { 
				// sample a user
				int u = rand.nextInt(userCount); 
				ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
				if (itemList.size() == 0)	continue;
				// sample a positive item
				int i = itemList.get(rand.nextInt(itemList.size())); 
				// sample a auxiliary item
				if (viewdata[u].length > 0) {
					int v = viewdata[u][rand.nextInt(viewdata[u].length)];
					// One SGD step update
					update_uiv(u, i, v);
				}
				else {
					update_ui(u, i);
				}
			}
			if ((iter >= maxIter - 10) || (iter % 10 == 0)) {
				if (validation) {
					showProgress(iter, start, validationRatings);
					for(Rating rating:validationRatings) {
						trainMatrix.setValue(rating.userId, rating.itemId, 1.0);
					}
					validationRatings.clear();
				}
				showProgress(iter, start, testRatings);
				if (adaptive && iter >= iter_adaptive) {
					double hr = hits.mean();
					if (hr < hr_prev) {
						lr = lr * 0.5;
						adaptive_flag = true;
					}
					else {
						lr = lr * 1.05;
						adaptive_flag = false;
					}
					hr_prev = hr;
				}
			}
//			else {
//				// Adjust the learning rate
//				if (adaptive_flag)
//					lr = lr * 0.5;
//			}
			// Show loss
			if (showLoss)
				loss_pre = showLoss(iter, start, loss_pre);
		}
		
	}

	//One SGD step for a positive instance.
	private void update_uiv(int u, int i, int v) {
		// sample a negative item (uniformly random)
		int j = rand.nextInt(itemCount);
		while (trainMatrix.getValue(u, j) != 0 || trainSideMatrix.getValue(u, j) != 0) {
			j = rand.nextInt(itemCount);
		}
		
		// update rules
		double y_pos = predict(u, i);  // target value of positive instance
		double y_aux = predict(u, v);  // target value of auxiliary instance
	    double y_neg = predict(u, j);  // target value of negative instance
	    double mult1 = 2 * (y_pos - 1);
	    double mult2 = 2 * y_neg;
	    double mult3 = 2 * (gamma1 - y_pos + y_aux);
	    double mult4 = 2 * (gamma2 - y_aux + y_neg);
	    
	    for (int f = 0; f < factors; f ++) {
	    	double grad_u1 = V.get(i, f);
	    	double grad_u2 = V.get(j, f);
	    	double grad_u3 = V.get(v, f);
	    	U.add(u, f, -lr * (mult1 * grad_u1 + mult2 * grad_u2 + 
	    			mult3 * (grad_u3 - grad_u1) + mult4 * (grad_u2 - grad_u3) +
	    			reg * U.get(u, f)));
	    	
	    	double grad = U.get(u, f);
	    	V.add(i, f, -lr * (mult1 * grad - mult3 * grad + reg * V.get(i, f)));
	    	V.add(v, f, -lr * (mult3 * grad - mult4 * grad + reg * V.get(v, f)));
	    	V.add(j, f, -lr * (mult2 * grad + mult4 * grad + reg * V.get(j, f)));
	    }
	}
	
	private void update_ui(int u, int i) {
		// sample a negative item (uniformly random)
		int j = rand.nextInt(itemCount);
		while (trainMatrix.getValue(u, j) != 0) {
			j = rand.nextInt(itemCount);
		}
		
		// update rules
		double y_pos = predict(u, i);  // target value of positive instance
		double y_neg = predict(u, j);  // target value of negative instance
	    double mult1 = 2 * (y_pos - 1);
	    double mult2 = 2 * y_neg;
	    
	    for (int f = 0; f < factors; f ++) {
	    	double grad_u1 = V.get(i, f);
	    	double grad_u2 = V.get(j, f);
	    	U.add(u, f, -lr * (mult1 * grad_u1 + mult2 * grad_u2 +
	    			reg * U.get(u, f)));
	    	
	    	double grad = U.get(u, f);
	    	V.add(i, f, -lr * (mult1 * grad + reg * V.get(i, f)));
	    	V.add(j, f, -lr * (mult2 * grad + reg * V.get(j, f)));
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

	@Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}

	
	@Override
	public void updateModel(int u, int i) {
		// TODO Auto-generated method stub
		
	}

}
