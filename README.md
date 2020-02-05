# Auxiliary_enhanced_ALS
The implementation of AALS

Auxiliary-enhanced eALS performs well by integrating users' multiple auxiliary data as the intermediate feedback. This is our official implementation for the paper: 

Jingtao Ding, Guanghui Yu, Yong Li, Xiangnan He, Depeng Jin and Jiajie Yu, **Improving Implicit Recommender Systems with Auxiliary Data**, To appear in ACM Transactions on Information Systems (TOIS).

If you use the codes, please cite our paper. Thanks!

# Requirements
JAVA

## Currently, for baselines, the eALS, BPR, MC-BPR, RankALS are provided. 

## The implementation of proposed pointwise and pairwise AALS methods are capable of incorporating one or two types of auxiliary data.

# Quick Start
## run Pair2AALS

java -jar <name_of_jar>.jar main_MF <path_of_primary_file> Pair2AALS <value_s0> False True <number_factors> <number_iterations> <reg_value> 0 <path_of_auxiliary1_file> <value_c0> 0 <value_gamma1> <value_gamma2> <path_of_auxiliary2_file> <value_c1> <value_gamma3> <flag_eval_topk> <validation_flag> 

## run Point2AALS

java -jar <name_of_jar>.jar main_MF <path_of_primary_file> Point2AALS <value_s0> False True <number_factors> <number_iterations> <reg_value> 0 <path_of_auxiliary1_file> <value_c0> 0 <value_r1> 0 <path_of_auxiliary2_file> <value_c1> <value_r2> <flag_eval_topk> <validation_flag>

## run Pair1AALS

java -jar <name_of_jar>.jar main_MF <path_of_primary_file> Pair1AALS <value_s0> False True <number_factors> <number_iterations> <reg_value> 0 <path_of_auxiliary1_file> <value_c0> 0 <value_gamma1> <value_gamma2> <validation_flag>

## run Point1AALS

java -jar <name_of_jar>.jar main_MF <path_of_primary_file> Point1AALS <value_s0> False True <number_factors> <number_iterations> <reg_value> 0 <path_of_auxiliary1_file> <value_c0> 0 <value_r1> 0 <validation_flag>

## run eALS

java -jar <name_of_jar>.jar main_MF <path_of_primary_file> fastals <value_s0> False True <number_factors> <number_iterations> <reg_value> 0

## run RankingALS

java -jar <name_of_jar>.jar main_MF <path_of_primary_file> rankingals 0 False True <number_factors> <number_iterations> <reg_value> 0 <path_of_auxiliary1_file> <value_r1> <path_of_auxiliary2_file> <value_r2>  <validation_flag>

## run BPR

java -jar <name_of_jar>.jar main_MF <path_of_primary_file> bpr <learning_rate> False True <number_factors> <number_iterations> <reg_value>

## run MC-BPR

java -jar <name_of_jar>.jar mfbpr_pos 0 0 <learning_rate> False True <number_factors> <number_iterations> <reg_value> 0 <path_of_primary_file> <path_of_auxiliary1_file> false <value_beta1> <value_beta2> 
