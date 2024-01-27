/**
 * Trained NaiveBayes Model
 * CostSensitiveClassifier that uses a NaiveBayes as the base classifier in Weka
 * To display the configuration settings of a CostSensitiveClassifier and the textual output of the classifier model in Weka, specifically for a CostSensitiveClassifier with a NaiveBayes base classifier
 * @author Mohammad Mustaneer Rahman
 */
import weka.classifiers.Classifier;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.SerializationHelper;
import weka.core.Utils;

public class ModelInspectorNB {
    public static void main(String[] args) {
        try {
            // The path to your NB_model_trained-2.model file
            String modelPath = "models/CostSensitive_NB_model_trained.model";

            // Deserialize the model from the file
            Classifier classifier = (Classifier) SerializationHelper.read(modelPath);

            // Check if the classifier is a CostSensitiveClassifier
            if (classifier instanceof CostSensitiveClassifier) {
                CostSensitiveClassifier costClassifier = (CostSensitiveClassifier) classifier;

                // Print the configuration settings of the CostSensitiveClassifier
                System.out.println("=== CostSensitiveClassifier Configuration ===");
                printClassifierOptions(costClassifier);

                // Get the base classifier and print its configuration if it is NaiveBayes
                Classifier baseClassifier = costClassifier.getClassifier();
                if (baseClassifier instanceof NaiveBayes) {
                    System.out.println("\n=== NaiveBayes Base Classifier Configuration ===");
                    printClassifierOptions((NaiveBayes) baseClassifier);
                }

                // Print the classifier model information
                System.out.println("\n=== Classifier Model ===");
                System.out.println(classifier.toString());
            } else {
                System.out.println("The loaded model is not a CostSensitiveClassifier.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void printClassifierOptions(Classifier classifier) {
        if (classifier instanceof NaiveBayes) {
            NaiveBayes naiveBayes = (NaiveBayes) classifier;
            System.out.println(Utils.joinOptions(naiveBayes.getOptions()));
        } else if (classifier instanceof CostSensitiveClassifier) {
            CostSensitiveClassifier costCls = (CostSensitiveClassifier) classifier;
            System.out.println(Utils.joinOptions(costCls.getOptions()));
        }
        // Include the cost matrix if available
        if (classifier instanceof CostSensitiveClassifier) {
            System.out.println("Cost Matrix: " + ((CostSensitiveClassifier) classifier).getCostMatrix().toString());
        }
    }
}


/* Note: 
CostSensitiveClassifier Configuration
Cost Matrix: The -cost-matrix "[0.0 1.0; 3.0 0.0]" specifies the cost of misclassifications between two classes. The costs are as follows:
0: Correct classifications incur no cost.
1: Misclassifying a class 0 instance as class 1 incurs a cost of 1.
3: Misclassifying a class 1 instance as class 0 incurs a higher cost of 3. This indicates that this type of error is considered more severe.
Seed (-S 1): This option sets the seed for random number generation to 1, which helps ensure reproducibility in the results.
Base Classifier (-W): Indicates that the NaiveBayes classifier is wrapped by the CostSensitiveClassifier.
NaiveBayes Base Classifier Configuration
This section is expected to list the configuration options for the NaiveBayes classifier, but none are provided in the text. These would normally include any parameters specific to how the NaiveBayes classifier operates.

Classifier Model
This section describes the model that has been built:

Classifier Type: The model is a CostSensitiveClassifier using reweighted training instances, which means it adjusts the weights of the instances during training according to the specified cost matrix to minimize the total cost of misclassifications.

Base Classifier: The NaiveBayes classifier is used as the underlying model. Naive Bayes classifiers are probabilistic classifiers based on applying Bayes' theorem with strong independence assumptions between the features.

Classifier Evaluation Metrics
The output includes a series of statistics for different attributes, broken down by the class prediction (no and yes):

Mean: The average value for each attribute for instances classified as no or yes.
Standard Deviation (std. dev.): Measures the amount of variation or dispersion of the attribute values from the mean.
Weight Sum: The total weight of instances that were used to calculate the statistics for the attribute.
Precision: This could refer to the precision of the attribute's mean estimate, but without further context, it's unusual to see precision reported this way for attribute values.
The numbers in parentheses next to the class labels (no and yes) indicate the prior probability of each class in the dataset. For example, (0.58) and (0.42) suggest that 58% of the instances are labeled as no and 42% as yes before any model training or prediction.

Cost Matrix
Reiterates the cost of misclassifications between classes, as described at the start. This is a crucial part of a cost-sensitive learning setup, emphasizing the asymmetric costs of different types of classification errors.

Interpretation of Results:
The model is configured to weigh false negatives (predicting no when the true class is yes) more heavily than false positives. This could reflect a domain-specific requirement where missing out on class yes instances is costlier.
The Naive Bayes Classifier is used to predict whether instances belong to class no or yes.
The provided statistics suggest how each attribute contributes to the classification decision for each class. Attributes that show a larger difference in means between classes may be more discriminative.
The CostSensitiveClassifier has modified the weight of the instances during training according to the cost matrix to reflect the different costs of misclassification.
This kind of output is useful for understanding how well different attributes help to distinguish between classes and how the cost matrix affects the learning process.

 * 
 * 
*/


