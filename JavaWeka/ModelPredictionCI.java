/**
 * To print results from Cost Sensitive Analysis 
 * @author Mohammad Mustaneer Rahman
 */

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.evaluation.Evaluation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class ModelPredictionCI {

    public static void main(String[] args) {
        try {
        /*Please load the models one by one */
            // Load the pre-trained classifier 
            String modelPath = "models/BN_model_Trained-2.model";
            
            // Load the pre-trained classifier NB
            //String modelPath = "models/NB_model_trained-2.model";

            // Load the pre-trained classifier j48
            //String modelPath = "models/J48_model_trained.model";

            // Load the pre-trained classifier RandomForest
            //String modelPath = "models/RandomForest_model_trained.model";

            // Load the pre-trained classifier KNN
            //String modelPath = "models/KNN_model_trained.model";

            // Load the pre-trained classifier BN Traditional
            //String modelPath = "models/BN_TRD_model_trained.model";

            // Load the pre-trained classifier NB Traditional
            //String modelPath = "models/NB_TRD_model_trained.model";
            
            Classifier classifier = (Classifier) SerializationHelper.read(modelPath);
            
        /*Please load test datasets one by one */
            // Load the input data UTAS dataset
            String inputPath = "data/TEST-UTAS-FINAL-424.arff";

            // Load the input data EmoDetect dataset
            //String inputPath = "data/TEST-EmoDetect-FINAL-300.arff";


            DataSource dataSource = new DataSource(inputPath);
            Instances inputData = dataSource.getDataSet();
            inputData.setClassIndex(inputData.numAttributes() - 1);

            // Evaluate the classifier on the test dataset
            Evaluation evaluation = new Evaluation(inputData);
            evaluation.evaluateModel(classifier, inputData);

            // Calculate the average AUC
            // https://weka.sourceforge.io/doc.dev/weka/classifiers/Evaluation.html
            // https://weka.sourceforge.io/doc.dev/weka/classifiers/Evaluation.html#areaUnderROC-int-

            double sumAUC = 0.0;
            int numClasses = inputData.numClasses();
            for (int i = 0; i < numClasses; i++) {
                double auc = evaluation.areaUnderROC(i);
                sumAUC += auc;
            }
            double averageAUC = sumAUC / numClasses;
            System.out.println("Average AUC: " + averageAUC);

            // Initialize parameters for bootstrapping
            int numBootstrapSamples = 1000;
            double confidenceLevel = 0.95;
            Random random = new Random();

            // Perform bootstrapping
            ArrayList<Double> aucList = new ArrayList<>();
            for (int b = 0; b < numBootstrapSamples; b++) {
                Instances resampledData = inputData.resample(random);
                Evaluation resampledEvaluation = new Evaluation(resampledData);
                resampledEvaluation.evaluateModel(classifier, resampledData);
                double resampledSumAUC = 0.0;
                for (int i = 0; i < numClasses; i++) {
                    double resampledAUC = resampledEvaluation.areaUnderROC(i);
                    resampledSumAUC += resampledAUC;
                }
                double resampledAverageAUC = resampledSumAUC / numClasses;
                aucList.add(resampledAverageAUC);
            }

            // Sort the AUCs
            Collections.sort(aucList);

            // Calculate the mean and standard deviation
            double sum = 0.0;
            for (double auc : aucList) {
                sum += auc;
            }
            double mean = sum / aucList.size();
            double sumOfSquares = 0.0;
            for (double auc : aucList) {
                double diff = auc - mean;
                sumOfSquares += diff * diff;
            }
            double variance = sumOfSquares / (aucList.size() - 1);
            double standardDeviation = Math.sqrt(variance);

            // Calculate the confidence intervals
            int lowerIndex = (int) (numBootstrapSamples * (1 - confidenceLevel) / 2);
            int upperIndex = (int) (numBootstrapSamples * (1 + confidenceLevel) / 2);
            double lowerCI = aucList.get(lowerIndex);
            double upperCI = aucList.get(upperIndex);

            // Print the results
            System.out.println("Sample Mean: " + mean);
            System.out.println("Standard Deviation: " + standardDeviation);
            System.out.println("Confidence Interval (" + (confidenceLevel * 100) + "%): [" + lowerCI + ", " + upperCI + "]");

            // Calculate the confidence intervals
            double halfWidth = (upperCI - lowerCI) / 2.0;
            System.out.printf("Confidence Interval (%.0f%%): %.2f ± %.2f\n", (confidenceLevel * 100), mean, halfWidth);


            // Calculate Cohen's kappa
            double tp = evaluation.weightedTruePositiveRate();
            double tn = evaluation.weightedTrueNegativeRate();
            double fp = evaluation.weightedFalsePositiveRate();
            double fn = evaluation.weightedFalseNegativeRate();
            double po = (tp + tn) / (tp + tn + fp + fn);
            double pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / Math.pow(tp + tn + fp + fn, 2);
            double kappa = (po - pe) / (1 - pe);
            System.out.println("Cohen's kappa: " + kappa);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
            
/* Note: 

Bootstrapping is used to estimate the uncertainty in the AUC estimate. 
Bootstrapping is a statistical technique for estimating the sampling distribution of an 
estimator by sampling from the data with replacement.

 * The confidence interval is calculated using the percentile method, 
 * which is a commonly used method for bootstrapping confidence intervals. 
 * Percentile method can be used for calculating confidence intervals 
 * for binary (two-class) classification problems.
 * The percentile method calculates the lower and upper bounds of the confidence interval 
 * as the (1 - confidenceLevel)/2 and (1 + confidenceLevel)/2 percentiles 
 * of the sorted bootstrapped samples, respectively.
 * 
 * 
 * areaUnderROC is a method in the Weka Evaluation class that calculates the area under the receiver operating 
 * characteristic (ROC) curve. The ROC curve is a graphical representation of the performance of a binary 
 * classifier as the discrimination threshold is varied. It plots the true positive rate (TPR) 
 * on the y-axis against the false positive rate (FPR) on the x-axis, where:

TPR = true positives / (true positives + false negatives)
FPR = false positives / (false positives + true negatives)
The area under the ROC curve (AUC) is a measure of the classifier's 
overall performance across all possible discrimination thresholds. An AUC of 0.5 indicates random performance, while an AUC of 1.0 indicates perfect performance.

In Weka, the areaUnderROC method calculates the AUC for a given class index. 
For example, if you have a binary classifier with two classes, 0 and 1, you can use areaUnderROC(0) 
to calculate the AUC for class 0, and areaUnderROC(1) to calculate the AUC for class 1.
 * 
 * 
*/