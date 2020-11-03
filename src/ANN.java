public class ANN {

    float[] input;
    float[] hidden;
    float[] output;

    float[][] weightsIH;
    float[][] weightsHO;

    float[] biasOutput;
    float[] biasHidden;
    float learningRate = 0.01f;


    public ANN(int input, int hidden, int output){

        this.input = new float[input];
        this.hidden = new float[hidden];
        this.output = new float[output];

        this.weightsIH = new float[hidden][input];
        this.weightsHO = new float[hidden][output];

        this.biasOutput = new float[output];
        this.biasHidden = new float[hidden];

        for (int i = 0; i < biasOutput.length; i++) {
                biasOutput[i] = (float)Math.random();
        }
        for (int i = 0; i < biasHidden.length; i++) {
                biasHidden[i] = (float)Math.random();
        }

        for(int row = 0; row < weightsIH.length; row++){
            for(int col = 0; col < weightsIH[row].length ; col++){
                weightsIH[row][col] = (float)Math.random();
            }
        }

        for(int row = 0; row < weightsHO.length; row++){
            for(int col = 0; col < weightsHO[row].length ; col++){
                weightsHO[row][col] = (float)Math.random()*2-1;
            }
        }

    }

    public void train(float[][] testCases, float[][] testSolutions,float margin, int iterations){

        float errorSum = margin+1;
        int iterCount = 0;
        while(iterCount < iterations){
            int picker = (int)Math.floor(Math.random()*(testCases.length));
            System.out.println("Test case "+testCases[picker][0]+" "+testCases[picker][1]+" "+testSolutions[picker][0]);
            errorSum = Math.abs(train(testCases[picker],testSolutions[picker]));

            iterCount++;
        }

        System.out.println("Weigths IH");
        for (float[] f: this.weightsIH) {
            for (float y: f) {
                System.out.println(y);
            }
        }
        System.out.println("Weigths HO");
        for (float[] f: this.weightsIH) {
            for (float y: f) {
                System.out.println(y);
            }
        }
        System.out.println("Training finished after "+iterCount+" iterations.");
        System.out.println("Error sum: "+errorSum);
        System.out.println("All weights updated!");
        
    }

    private float train(float[] input, float[] solution){

        float[] results = predict(input);
        float[] derivSigmoid = derivSigmoid(results.clone());
        float[] derivSigmoidHidden = derivSigmoid(hidden.clone());
        float[] error = subtract(solution,results);
        float[] costHidden = new float[hidden.length];
        for (int i = 0; i < hidden.length; i++) {
            for (int j = 0; j < results.length; j++) {
                costHidden[i] += weightsHO[i][j]*(-error[j]);
            }
        }


        for (int i = 0; i < biasOutput.length; i++) {
            biasOutput[i] -= 1 * derivSigmoid[i]* (-error[i]) * learningRate;
        }
        for (int i = 0; i < biasHidden.length; i++) {
            biasHidden[i] -= 1 * derivSigmoidHidden[i] * costHidden[i] * learningRate;
        }



        for (int i = 0; i < weightsHO.length; i++) {
            for (int z = 0; z < weightsHO[0].length; z++) {
                weightsHO[i][z] -= (hidden[i] * derivSigmoid[z] * (-error[z]) * learningRate);
            }
        }
        for (int i = 0; i < weightsIH[0].length; i++) {
            for (int z = 0; z < weightsIH.length; z++) {
                weightsIH[z][i] -= (input[i] * costHidden[z] * derivSigmoidHidden[z] * learningRate);
            }
        }


        return 1;
    }

    public float[] predict(float[] input){
        this.input = input;

        calculateNeurons();
        activateNeurons();

        calculateOutput();
        activateOutputs();

        return this.output;

    }

    private void calculateNeurons(){
        for(int neuron = 0; neuron < hidden.length; neuron++){
            hidden[neuron] = 0.0f;
            for(int in = 0; in < input.length; in++){
                hidden[neuron] += input[in] * weightsIH[neuron][in];
            }
            hidden[neuron] += biasHidden[neuron];
        }

    }

    private void activateNeurons(){
        for(int neuron = 0; neuron < hidden.length; neuron++){
            hidden[neuron] = (float)(1/(1+Math.exp(-hidden[neuron])));
        }
    }

    private void calculateOutput(){
        for(int out = 0; out < output.length; out++){
            output[out] = 0.0f;
            for(int neuron = 0; neuron < hidden.length; neuron++){
                output[out] += hidden[neuron] * weightsHO[neuron][out];
            }
            output[out] += biasOutput[out];
        }
    }

    private void activateOutputs(){
        for(int out = 0; out < output.length; out++){
            output[out] = (float)(1/(1+Math.exp(-output[out])));
        }
    }

    private static float[] derivSigmoid(float[] arg1){
        for(int i = 0; i < arg1.length; i++){
            arg1[i]= arg1[i]*(1-arg1[i]);
        }
        return arg1;
    }


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



    public static void main(String[] args) {

        ANN ann = new ANN(2,3,1);
        float[][] trainData = {{0,0},
                {1,0},
                {0,1},
                {1,1}};
        float[][] trainSolutions = {{0},{1},{1},{0}};

        ann.train(trainData,trainSolutions,0.001f,500000);

        float[] inputs = new float[]{0,0};
        float[] result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (0,0): "+res);
        }
        inputs = new float[]{1,0};
        result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (1,0): "+res);
        }
        inputs = new float[]{0,1};
        result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (0,1): "+res);
        }
        inputs = new float[]{1,1};
        result = ann.predict(inputs);
        for(Float res: result){
            System.out.println("Result (1,1): "+res);
        }

    }

}
