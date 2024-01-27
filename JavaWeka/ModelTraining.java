/**
 * To print results from Cost Sensitive Analysis
 * Performance evaluation of two cost-sensitive Bayesian variants, 
 * trained on the resampled dataset
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
 
 public class ModelTraining {
 
     public static void main(String[] args) {
         try {
         /*Please load the models one by one */
              // Load the pre-trained classifier 
              String modelPath = "models/CostSensitive_BN_model_Trained.model";
             
              // Load the pre-trained classifier NB
              //String modelPath = "models/CostSensitive_NB_model_trained.model";

             
             Classifier classifier = (Classifier) SerializationHelper.read(modelPath);
             
         /*Please load datasets ASSISTments Sampled and Original one by one.
          * Trained on Resampled ASSISTments Dataset */
         /* Load the input data */
            
            // Load the input data Resampled ASSISTments dataset 12% “Yes”; Instances: 2130
            //String inputPath = "data/Training/Affect_Clip_Labels_important-7-features_Resampled_12%.arff";
         
            // Load the input data Resampled ASSISTments dataset 20% “Yes”; Instances: 2284
            //String inputPath = "data/Training/Affect_Clip_Labels_important-7-features_Resampled_20%.arff";
       
            // Load the input data Resampled ASSISTments dataset 30% “Yes”; Instances: 2658
            //String inputPath = "data/Training/Affect_Clip_Labels_important-7-features_Resampled_30%.arff";
        
            /* Validation on Non-Resampled ASSISTments Dataset */
            // Load the input data ASSISTments original dataset 4% “Yes”; Instances: 2633
            String inputPath = "data/Training/Affect_Clip_Labels_important-7-features.arff";
             
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
            System.err.println("\n");
            System.err.println("----------Cost Sensitive Analysis----------");
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

        
            // Print the confusion matrix with proper alignment and labels
            System.out.println("=== Confusion Matrix ===");
            double[][] confusionMatrix = evaluation.confusionMatrix();

            // Print header
            System.out.println("\ta\tb\t<- classified as");
            // Print each row of the confusion matrix
            for (int i = 0; i < confusionMatrix.length; i++) {
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    System.out.print("\t" + (int) confusionMatrix[i][j]);
                }
                // Print the row label
                if (i == 0) {
                    System.out.println("\t| a = no");
                } else {
                    System.out.println("\t| b = yes");
                }
            }

            // Calculate Cohen's Kappa
            double totalCorrect = 0.0;
            double totalByChance = 0.0;
            double totalInstances = inputData.numInstances();

            for (int i = 0; i < numClasses; i++) {
                totalCorrect += confusionMatrix[i][i];
                double rowSum = 0.0;
                double colSum = 0.0;
                for (int j = 0; j < numClasses; j++) {
                    rowSum += confusionMatrix[i][j];
                    colSum += confusionMatrix[j][i];
                }
                totalByChance += (rowSum * colSum);
            }

            double po = totalCorrect / totalInstances;
            double pe = totalByChance / (totalInstances * totalInstances);
            double kappa = (po - pe) / (1 - pe);

            System.out.println("\nCohen's Kappa: " + kappa);


           // Calculate and print Precision, Recall, and F-Measure for each class
            System.out.println("\n=== Precision, Recall, and F-Measure for Each Class ===");
            double totalFMeasure = 0.0;
            for (int i = 0; i < numClasses; i++) {
                double precision = evaluation.precision(i);
                double recall = evaluation.recall(i);
                double fMeasure = evaluation.fMeasure(i);
                System.out.println("Class " + i + ":");
                System.out.println("Precision: " + precision);
                System.out.println("Recall: " + recall);
                System.out.println("F-Measure: " + fMeasure);
                totalFMeasure += fMeasure;
            }

            // Calculate and print average F-Measure
            double averageFMeasure = totalFMeasure / numClasses;
            System.out.println("\n=== Average F-Measure ===");
            System.out.println("Average F-Measure: " + averageFMeasure);
            System.out.println();


            // Calculate and print Weighted Average AUC
            double weightedAUC = 0.0;
            for (int i = 0; i < numClasses; i++) {
                double classProportion = (double) inputData.attributeStats(inputData.classIndex()).nominalCounts[i] / inputData.numInstances();
                double classAUC = evaluation.areaUnderROC(i);
                weightedAUC += classAUC * classProportion;
            }
            System.out.println("Weighted Average AUC: " + weightedAUC);

            // Calculate and print weighted Precision, Recall, and F-Measure
            double weightedPrecision = evaluation.weightedPrecision();
            double weightedRecall = evaluation.weightedRecall();
            double weightedFMeasure = evaluation.weightedFMeasure();

            System.out.println("Weighted Average Precision: " + weightedPrecision);
            System.out.println("Weighted Average Recall: " + weightedRecall);
            System.out.println("Weighted Average F-Measure: " + weightedFMeasure);

            System.out.println("\n");
            
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