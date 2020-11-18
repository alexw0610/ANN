
public class Main {
    /**
     * A quick demo of the neural net implementation using two input neurons and one output neuron.
     * The example that the net learns is the XOR truth table.
     * @param args none
     */
    public static void main(String[] args) {

        ANN ann = new ANN(2,10,10,1);
        float[][] trainData = {{0,0},
                {1,0},
                {0,1},
                {1,1}};
        float[][] trainSolutions = {{0},{1},{1},{0}};

        Display display = new Display(1024,768);

        display.canvas.reshape(0,0,1024,768);
        display.frame.repaint();
        display.displayNet(ann);

        for(int cycle = 0; cycle <100000;cycle++){
            ann.train(trainData,trainSolutions,1,0.01f,1);
            display.displayNet(ann);
            try {
                Thread.sleep(250);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }




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
