class XOR{
    /**
     * A quick demo of the neural net implementation using two input neurons and one output neuron.
     * The example that the net learns is the XOR truth table.
     * @param args none
     */
    public static void main(String[] args) {

        ANN ann = new ANN(2,10,5,1).learningRate(0.02f);

        float[][] trainData = {
                {0,0},
                {1,0},
                {0,1},
                {1,1}
        };

        float[][] trainSolutions = {{0},{1},{1},{0}};

        ann.train(trainData,trainSolutions,50000,0.01f,3);

        /*
        Persist the weights of a trained model to disk inorder to load and reuse them later.
                ann.persistWeightsToDisk("./ANN.model");
                ann.loadWeightsFromDisk("./ANN.model");
        */


        float[] inputs = new float[]{0,0};
        float[] result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (0,0): "+Math.round(res)+" was "+res);
        }
        inputs = new float[]{1,0};
        result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (1,0): "+Math.round(res)+" was "+res);
        }
        inputs = new float[]{0,1};
        result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (0,1): "+Math.round(res)+" was "+res);
        }
        inputs = new float[]{1,1};
        result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (1,1): "+Math.round(res)+" was "+res);
        }

    }
}