/**
 * Predictions for each instances
 * @author Mohammad Mustaneer Rahman
 */

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class ModelPrediction {
// Using the trained models classifying instances on the test dataset
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

            // Set the class attribute (assuming the last attribute is the class)
            inputData.setClassIndex(inputData.numAttributes() - 1);
            
            // Make predictions for each instance in the input data
            for (Instance instance : inputData) {
                //System.out.print(instance);
                double prediction = classifier.classifyInstance(instance);
                String predictedClassLabel = inputData.classAttribute().value((int) prediction);
                System.out.println("Predicted class label: " + predictedClassLabel);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}