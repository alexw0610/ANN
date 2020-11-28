import java.io.*;
import java.nio.ByteBuffer;

public class ANN {

    public float[] input;
    public final float[][] hidden;
    public float[] output;

    public float[][][] weights;
    public final float[][] bias;

    public float learningRate = 0.02f;


    public ANN(int input, int hidden, int layer, int output){

        /*  |      <----       layer       ---->     |
        *               0hidden ... 0hidden
        *   0input  -   0hidden ... 0hidden   -   0output
        *      .        0hidden ... 0hidden           .
        *      .            .          .              .
        * */

        this.input = new float[input];
        this.hidden = new float[layer-2][hidden];
        this.output = new float[output];

        float[][] weightsIH = new float[input][hidden];
        float[][] weightsHO = new float[hidden][output];
        this.weights = new float[layer-1][][];

        this.bias = new float[layer-1][];

        /*
          Fill bias weights with random values.
         */

        for (int i = 0; i < this.bias.length; i++) {
                if(i < this.bias.length-1){
                    float[] temp = new float[this.hidden[0].length];
                    for (int j = 0; j < temp.length; j++) {
                        temp[j] = (float)Math.random()*2-1;
                    }
                    this.bias[i] = temp;
                }else{
                    float[] temp = new float[this.output.length];
                    for (int j = 0; j < temp.length; j++) {
                        temp[j] = (float)Math.random()*2-1;
                    }
                    this.bias[i] = temp;
                }
        }


        /*
          Fill input to first hidden layer weights with random values.
         */
        for(int row = 0; row < weightsIH.length; row++){
            for(int col = 0; col < weightsIH[row].length ; col++){
                weightsIH[row][col] = (float)Math.random()*2-1;
            }
        }

        /*
          Fill last hidden layer to output weights with random values.
         */
        for(int row = 0; row < weightsHO.length; row++){
            for(int col = 0; col < weightsHO[row].length ; col++){
                weightsHO[row][col] = (float)Math.random()*2-1;
            }
        }

        /*
          Place first and last weight sets into 3d-weight array.
         */
        this.weights[0] = weightsIH;
        this.weights[this.weights.length-1] = weightsHO;

        /*
          Fill and place intermediate weight sets into 3d-weight array.
         */
        for (int layerIndex = 1; layerIndex < this.weights.length-1; layerIndex++) {
            float[][] temp = new float[hidden][hidden];
            for(int row = 0; row < hidden; row++){
                for(int col = 0; col < hidden ; col++){
                    temp[row][col] = (float)Math.random()*2-1;
                }
            }
            this.weights[layerIndex]  = temp;
        }

    }

    /**
     * Set the desired learning rate of the neural network.
     * @param rate a non negative float value representing the desired learning rate
     * @return the instance of ANN with an updated learning rate
     */
    public ANN learningRate(float rate) {
        if(rate >= 0.0f){
            this.learningRate = rate;
            return this;
        }else{
            System.err.println("Error: learning rate can not be negative. Default was used.");
            return this;
        }
    }


    /**
     * The exposed train function that handles each iteration of training
     * @param testCases a 2d array containing arrays with input values
     * @param testSolutions a 2d array containing arrays with solutions mapping to the input values
     * @param iterations the desired iterations for which the training should be performed
     * @param margin the margin value that the mean squared error of the network needs to beat NOTE: not used yet
     * @param verbosityLevel the desired level of information printed to the console
     */
    public void train(float[][] testCases, float[][] testSolutions, int iterations, float margin, int verbosityLevel){

        float errorSum = 0;
        int iterCount = 0;
        int[] testCasesCounter = new int[testCases.length];

        while(iterCount < iterations){
            int picker = (int)Math.floor(Math.random()*(testCases.length));
            testCasesCounter[picker] += 1;
            errorSum = train(testCases[picker],testSolutions[picker]);
            iterCount++;
        }

        if(verbosityLevel>1){
            System.out.println("Test cases distribution:");
            for (int i = 0; i < testCasesCounter.length; i++) {
                System.out.println("Test case #"+i+" : "+testCasesCounter[i]+" times seen during training!");
            }
        }

        if(verbosityLevel>0){
            System.out.println("Training finished after "+iterCount+" iterations.");
            if(errorSum < margin){
                System.out.println("Target error of "+margin+" was reached. error@lastIteration: "+errorSum);
            }else{
                System.out.println("error@lastIteration:: "+errorSum);
            }
        }

    }

    /**
     * The training function that updates the weight matrix depending on the input,the supplied solution for the input and the set learning rate
     * @param input The input for which the network weights should be adjusted
     * @param solution The solution to the input set from which the error or difference will be calculated
     * @return the mean squared error of the networks result compared to the supplied solution
     */
    private float train(float[] input, float[] solution){

        float[] results = predict(input);
        float[] costTotal = subtract(solution,results);
        float[] derivSigmoid = derivSigmoid(results.clone());
        float[][][] weightsUpdated = weights.clone();

        // weights from last hidden to output layer
        for (int weightFrom = 0; weightFrom < weights[weights.length-1].length; weightFrom++) {
            for (int weightTo = 0; weightTo < weights[weights.length-1][weightFrom].length; weightTo++) {
                weightsUpdated[weights.length-1][weightFrom][weightTo] -= hidden[hidden.length-1][weightFrom] * derivSigmoid[weightTo] * (-costTotal[weightTo]) * learningRate;
            }
        }
        //bias weights for output neurons
        for (int i = 0; i < bias[bias.length-1].length; i++) {
            bias[bias.length-1][i] -= 1 * derivSigmoid[i] * (-costTotal[i]) * learningRate;
        }



        // all weights in between hidden layers
        float[] errorNextLayer = costTotal.clone();
        for (int weightLayer = weights.length-2; weightLayer > 0; weightLayer--) {

            //calculate the deriv of the next layer nodes
            derivSigmoid = derivSigmoid(hidden[weightLayer].clone());

            //calculate the error in respect to of the next layer nodes
            float[] error = new float[weights[weightLayer+1].length];
            for (int i = 0; i < error.length; i++) {
                for (int j = 0; j < weights[weightLayer+1][0].length; j++) {
                    error[i] += weights[weightLayer+1][i][j]*(errorNextLayer[j]);
                }
            }

            //normalize the error to avoid gradient explosion
            errorNextLayer = normalize(error);

            //calculate the weight from this layer to the next
            for (int weightFrom = 0; weightFrom < weights[weightLayer].length; weightFrom++) {
                for (int weightTo = 0; weightTo < weights[weightLayer][weightFrom].length; weightTo++) {
                    weightsUpdated[weightLayer][weightFrom][weightTo] -= hidden[weightLayer-1][weightFrom] * derivSigmoid[weightTo] *(-errorNextLayer[weightTo]) * learningRate;
                }
            }

            //calculate the bias weights for this layers neurons
            for (int i = 0; i < bias[weightLayer].length; i++) {
                bias[weightLayer][i] -= 1 * derivSigmoid[i] * (-errorNextLayer[i]) * learningRate;
            }

        }


        // weights from input to first hidden layer
        derivSigmoid = derivSigmoid(hidden[0].clone());
        float[] error = new float[weights[1].length];
        for (int i = 0; i < error.length; i++) {
            for (int j = 0; j < weights[1][0].length; j++) {
                error[i] += weights[1][i][j]*(errorNextLayer[j]);
            }
        }
        //normalize the error to avoid gradient explosion
        error = normalize(error);
        for (int weightFrom = 0; weightFrom < weights[0].length; weightFrom++) {
            for (int weightTo = 0; weightTo < weights[0][weightFrom].length; weightTo++) {
                weightsUpdated[0][weightFrom][weightTo] -= input[weightFrom] * derivSigmoid[weightTo] * (-error[weightTo]) * learningRate;
            }
        }



        /*
            weights from this layer -= neuron from current layer
                                     * derivSigmoid(next layer)
                                     * -sum((weights from this neuron to next layer) * (cost from next layer))
                                     * LEARNING_RATE
        */

        //update the all the weights after the iteration
        this.weights = weightsUpdated;


        return sum(meanSquared(costTotal));
    }

    public float[] predict(float[] input){
        this.input = input;
        /*
         * Calculate and activate the first hidden neuron layer (from input to neuron)
         */
        float[] tempHidden = calculateNeurons(this.input,weights[0],this.bias[0]);
        this.hidden[0] = activateNeurons(tempHidden);

        /*
         * Calculate and activate all other hidden neuron layers
         */
        for (int i = 1; i < this.hidden.length; i++) {
            this.hidden[i] = calculateNeurons(this.hidden[i-1],weights[i],this.bias[i]);
            this.hidden[i] = activateNeurons(this.hidden[i]);
        }

        /*
         * Calculate and activate the output layer
         */
        float[] tempOutput = calculateNeurons(this.hidden[this.hidden.length-1],weights[this.weights.length-1],this.bias[this.bias.length-1]);
        this.output = activateNeurons(tempOutput);

        return this.output;
    }


    /**
     * Calculates the values each neuron holds.
     * The value is the sum of all incoming weights multiplied with their origin neurons value together with the bias for the neuron
     * @param input The origin neurons
     * @param weights The weight matrix containing the weights for each edge between two neurons
     * @param bias The bias array containing the bias values for each neuron
     * @return  The array of neurons with their values set
     */
    private float[] calculateNeurons(float[] input, float[][] weights, float[] bias){
        float[] temp = new float[weights[0].length];
        for(int neuron = 0; neuron < temp.length; neuron++){
            temp[neuron] = 0.0f;
            for(int in = 0; in < input.length; in++){
                temp[neuron] += input[in] * weights[in][neuron];
            }
            temp[neuron] += bias[neuron];
        }
        return temp;

    }


    /**
     * Applies the activation function to each neurons value
     * @param input The array of neurons that need to be activated
     * @return  The activated array of neurons
     */
    private float[] activateNeurons(float[] input){
        for(int neuron = 0; neuron < input.length; neuron++){
            input[neuron] = (float)(1/(1+Math.exp(-input[neuron])));
        }
        return input;
    }


    /**
     * calculates the derivative of the sigmoid function and applies the operation to the input values
     * @param arg1 the array containing the values on which the deriviative sigmoid function should be applied
     * @return an array containing the results of the operation
     */
    private static float[] derivSigmoid(float[] arg1){
        for(int i = 0; i < arg1.length; i++){
            arg1[i]= arg1[i]*(1-arg1[i]);
        }
        return arg1;
    }


    /**
     * pairwise subtraction of the values in two equal lengths arrays
     * @param arg1 the first array containing the first element of each subtraction
     * @param arg2 the second array containing the second element of each subtraction
     * @return an array containing the calculated differences
     */
    private static float[] subtract(float[] arg1, float[] arg2){

        if(arg1.length != arg2.length){
            System.err.println("Subtraction error. Arrays are not of equal length!");
        }

        float[] temp = new float[arg1.length];
        for(int i = 0; i < arg1.length; i++){
            temp[i] = arg1[i]-arg2[i];
        }
        return temp;
    }


    /**
     * calculates the total sum of all the entires of an array.
     * @param input the array containing the elements to be summed up
     * @return the sum of the element as a floating point value
     */
    private static float sum(float[] input){
        float result = 0;
        for(int i = 0; i < input.length;i++){
            result += Math.abs(input[i]);
        }
        return result;
    }


    /**
     * calculates the the mean squared value of each array input
     * NOTE: this is a helper function that relies on the input already being the difference of two values.
     * usually the mean squared function is defined as follows:
     *          meanSquared(x,y) = (1/2)*(|x-y|^2)
     * @param input an array containing differences between two values
     * @return the mean squared value of the difference
     */
    private static float[] meanSquared(float[] input){
        for (int i = 0; i < input.length; i++) {
            input[i] = (float)(Math.pow(input[i],2)*0.5f);
        }
        return input;
    }

    /**
     * normalizes the array of float to values between [-1,1]
     * @param input the array containing the values to normalize
     * @return an array containing the normalized values
     */
    private static float[] normalize(float[] input){
        float[] temp = new float[input.length];
        float biggestValue = 0;
        for(int index = 0; index < input.length; index++) {
            biggestValue = biggestValue > Math.abs(input[index]) ? biggestValue : Math.abs(input[index]);
        }
        for (int index = 0; index < input.length; index++) {
            temp[index] = input[index]/biggestValue;
        }
        return temp;
    }


    /**
     * persists the current weights and biases from the model to disk for later reuse.
     * @param path the path to the desired .model file location
     */
    public void persistWeightsToDisk(String path){

        File file = new File(path);

        try {
            FileOutputStream out = new FileOutputStream(file, false);
            int count = 0;
            for (int layer = 0; layer < this.weights.length; layer++) {
                for (int from = 0; from < this.weights[layer].length; from++) {
                    for (int to = 0; to < this.weights[layer][from].length; to++) {
                        count++;
                        out.write(floatToByteArray(this.weights[layer][from][to]));
                    }
                }
            }
            for (int biasLayer = 0; biasLayer < this.bias.length; biasLayer++) {
                for (int biasWeight = 0; biasWeight < this.bias[biasLayer].length; biasWeight++) {
                    count++;
                    out.write(floatToByteArray(this.bias[biasLayer][biasWeight]));
                }
            }

            out.close();
            System.out.println("Saved "+(count*4)+" bytes to "+path);

        } catch (IOException e) {
            System.err.println("Error failed to persist model to disk!");
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("Error while loading the weights from a file. \n" +
                    "The .model file doesn't fit the current model settings [input hidden layer output] dimensions might be different");
        }



    }

    /**
     * loads saved weights from a .model file to continue training or predict input
     * WARNING: This method expects the exact order of weights as bytes and performs no safety checks.
     * If the chosen .model file throws an Exception the weight file might have been generated for a model with different dimensions.
     * If the .model file was generated by a model with different dimensions and no error is thrown the prediction results will most likely be nonsense.
     * @param path to the .model file
     */
    public void loadWeightsFromDisk(String path){

        File file = new File(path);

        if(file.exists()){
            try {
                FileInputStream in = new FileInputStream(file);
                int count = 0;
                for (int layer = 0; layer < this.weights.length; layer++) {
                    for (int from = 0; from < this.weights[layer].length; from++) {
                        for (int to = 0; to < this.weights[layer][from].length; to++) {
                            count++;
                            this.weights[layer][from][to] = byteArrayToFloat(in.readNBytes(4));
                        }
                    }
                }
                for (int biasLayer = 0; biasLayer < this.bias.length; biasLayer++) {
                    for (int biasWeight = 0; biasWeight < this.bias[biasLayer].length; biasWeight++) {
                        count++;
                        this.bias[biasLayer][biasWeight] = byteArrayToFloat(in.readNBytes(4));

                    }
                }

                in.close();
                System.out.println("Loaded "+(count*4)+" bytes from "+path);
                System.out.println("Updated "+count+" weights");

            } catch (IOException e) {
                System.err.println("Error failed to persist model to disk!");
            }
        }

    }

    /**
     * stores the bytes of the float into an array
     * @param input the float that needs to be stored
     * @return a 4 byte array containing the representation of the float
     */
    private byte[] floatToByteArray(float input){
        return ByteBuffer.allocate(4).putFloat(input).array();
    }


    /**
     * converts a 4 byte array into a floating point number
     * @param input the byte array containing the bytes of the float
     * @return a floating point representation of the bytes
     */
    private float byteArrayToFloat(byte[] input){
        return ByteBuffer.wrap(input).getFloat();
    }




}
