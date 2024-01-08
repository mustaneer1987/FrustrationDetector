/**
 * To print configuration of the seven Weka classifiers together.
 * @author Mohammad Mustaneer Rahman
 */
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

public class ModelInspector {
    public static void main(String[] args) {
        try {
            // Array of model paths
            String[] modelPaths = {
                "models/BN_model_Trained-2.model",
                "models/NB_model_trained-2.model",
                "models/J48_model_trained.model",
                "models/RandomForest_model_trained.model",
                "models/KNN_model_trained.model",
                "models/BN_TRD_model_trained.model",
                "models/NB_TRD_model_trained.model"
            };

            // Iterate over each model path
            for (String modelPath : modelPaths) {
                // Deserialize the model from the file
                Classifier model = (Classifier) SerializationHelper.read(modelPath);

                // Print the model name and its evaluation result
                System.out.println(modelPath + " Result:");
                System.out.println(inspectModel(model));
                System.out.println(); // Print an empty line between models' results
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    // Method to inspect the model and return a string representation of the result
    private static String inspectModel(Classifier model) {
        // Depending on the type of model, we may need to cast it to the appropriate class
        // and call specific methods to get the details you are interested in.
        // For simplicity, we just call toString() on the model for this example.
        return model.toString();
    }
}

/* Note: 
It is not possible to see the source code of a saved model in Java because the saved file contains 
a serialized object representing the state of the model, not the actual Java source code. 
Serialization captures the data within an object at a specific moment in time and is intended 
for storage or transmission, not for human reading or editing. Java source code, on the other hand, 
is written in text files with .java extensions and compiled into bytecode, 
which is what runs on the Java Virtual Machine (JVM). 
The serialized model file only includes information about the model's parameters and 
learned data necessary to make predictions, not the algorithms (source code) that created the model.
 * 
 * 
*/