from data import trainingSet, testSet, outputDict, inputDim, outputDim, hiddenDim
from node import nodeModel 

class linkData:
    def __init__(self):
        def flatten(matrix):
            return [v for row in matrix for v in row]

        X_train = [flatten(inp) for inp, label in trainingSet]
        Y_train = [outputDict[label] for inp, label in trainingSet]

        X_test = [flatten(inp) for inp, label in testSet]
        Y_test = [outputDict[label] for inp, label in testSet]

        model = nodeModel(inputDim, outputDim, hiddenDim)
        lst_loss = model.train(X_train, Y_train, epochs=100, lr=0.1)

        set_as_label = {0: 'O', 1: 'X'}
        
        # Use the prediction method that returns (class, confidence)
        predictions = model.predict(X_test)

        model.draw_result(lst_loss, "MSEloss")



        print("Test Results:")
        correct = 0
        for i, ((pred_class, confidence), true_vec) in enumerate(zip(predictions, Y_test)):
            true_class = true_vec.index(1)
            is_correct = pred_class == true_class
            if is_correct:
                correct += 1
            print(f"Sample {i+1}: Predicted = {set_as_label[pred_class]}, Confidence = ({confidence*100:.2f}%), "
                  f"Actual = {set_as_label[true_class]} {'true' if is_correct else 'false'}")

        accuracy = correct / len(Y_test)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        
