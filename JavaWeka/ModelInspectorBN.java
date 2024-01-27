/**
 * Trained Bayesian Network Model
 * CostSensitiveClassifier using BayesNet as the base classifier in Weka
 * To display the configuration settings of a CostSensitiveClassifier and the textual output of the classifier model in Weka, specifically for a CostSensitiveClassifier with a BayesNet base classifier
 * @author Mohammad Mustaneer Rahman
 */
import weka.classifiers.Classifier;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.classifiers.bayes.BayesNet;

public class ModelInspectorBN {
    public static void main(String[] args) {
        try {
            // Path to the Weka model file
            String modelPath = "models/CostSensitive_BN_model_Trained.model";

            // Deserialize the model from the file
            Classifier classifier = (Classifier) SerializationHelper.read(modelPath);

            // Check if the classifier is a CostSensitiveClassifier
            if (classifier instanceof CostSensitiveClassifier) {
                CostSensitiveClassifier costClassifier = (CostSensitiveClassifier) classifier;

                // Print the classifier details
                System.out.println("=== Classifier Configuration ===");
                System.out.println("CostSensitiveClassifier options:");
                printOptions(costClassifier);

                // Print the cost matrix
                System.out.println("Cost Matrix:");
                System.out.println(costClassifier.getCostMatrix().toString());

                // Print the base classifier details if it is a BayesNet
                Classifier baseClassifier = costClassifier.getClassifier();
                if (baseClassifier instanceof BayesNet) {
                    System.out.println("Base Classifier (BayesNet) options:");
                    printOptions((BayesNet) baseClassifier);
                    
                    // Print the textual model of the BayesNet
                    System.out.println("=== Classifier Model ===");
                    System.out.println(baseClassifier.toString());
                }
            } else {
                System.out.println("The loaded model is not a CostSensitiveClassifier.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void printOptions(Classifier classifier) throws Exception {
        if (classifier instanceof BayesNet) {
            BayesNet bayesNet = (BayesNet) classifier;
            String[] options = bayesNet.getOptions();
            System.out.println(Utils.joinOptions(options));
        } else if (classifier instanceof CostSensitiveClassifier) {
            CostSensitiveClassifier costCls = (CostSensitiveClassifier) classifier;
            String[] options = costCls.getOptions();
            System.out.println(Utils.joinOptions(options));
        }
    }
}

/* Note: 

Classifier Configuration
CostSensitiveClassifier options: This line shows the configured options for the CostSensitiveClassifier. It includes:

Cost Matrix: Specified as "[0.0 1.0; 3.0 0.0]", which is a 2x2 matrix representing the cost of misclassifications. The matrix suggests that there is no cost for correctly classified instances (0 cost for true positives and true negatives), a cost of 1 for false negatives (misclassifying class 0 as class 1), and a higher cost of 3 for false positives (misclassifying class 1 as class 0).
Seed (-S 1): The seed for randomization processes within the classifier is set to 1, which helps in replicating the results.
Base Classifier (-W): Indicates that the BayesNet classifier is used as the base classifier for the CostSensitiveClassifier.
Cost Matrix: Presented again in a simpler format, reaffirming the costs for misclassifications.

Base Classifier (BayesNet) options
These are the options set specifically for the BayesNet classifier, which include:

-D: This likely indicates that the classifier should not use the ADTree data structure.
Search Algorithm (-Q): The K2 local search algorithm is used for learning the structure of the BayesNet.
Initial Structure (-P 1): Indicates starting with a structure akin to Naive Bayes.
Scoring Type (-S BAYES): Uses a scoring algorithm based on Bayesian probability.
Estimator (-E): The SimpleEstimator is used with an alpha value of 0.5, which might be used for smoothing probabilities.
Classifier Model
Bayes Network Classifier: This indicates the type of the classifier model used.
Attributes and Class Index: The model uses 8 attributes (features), with the 7th attribute being used as the class label.
Network Structure: This outlines the structure of the Bayesian network, including the nodes (attributes) and their parents (dependencies). For example, averagecorrect(15): class suggests that the averagecorrect node is directly dependent on the class node.
LogScore Bayes, BDeu, MDL, ENTROPY, AIC: These are various scoring functions used to evaluate the fit of the Bayesian network to the data:
LogScore Bayes: The log likelihood of the data given the Bayesian network structure.
LogScore BDeu: Bayesian Dirichlet equivalent uniform score.
LogScore MDL: Minimum Description Length, a score based on information theory.
LogScore ENTROPY: A score based on entropy, reflecting the uncertainty in the data.
LogScore AIC: Akaike Information Criterion, which balances model fit and complexity.
Interpretation of the Results:
The CostSensitiveClassifier has been set up to penalize false positives more than false negatives, which suggests that in the context of this classifier, predicting class 1 when it's actually class 0 is considered a more severe error than the opposite.
The BayesNet structure and its parameters indicate that the model is looking for dependencies between different attributes and the class label.
The scores (LogScore Bayes, BDeu, etc.) indicate the model's performance. Lower scores are generally better, indicating a higher likelihood or lower complexity, depending on the score type.
The specific values of the log scores are negative, which is common because these scores are usually derived from negative log-likelihoods.
The exact interpretation of the log scores would depend on comparisons to other models or scores on the same dataset.
 * 
 * 
*/