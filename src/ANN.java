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

    public void train(float[][] testCases, float[] testSolutions,float margin, int iterations){
        float errorSum = margin+1;
        int iterCount = 0;
        while(iterCount < iterations){
            int picker = (int)Math.floor(Math.random()*(testCases.length));
            System.out.println("Test case "+testCases[picker][0]+" "+testCases[picker][1]+" "+testSolutions[picker]);
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

    private float train(float[] input, float solution){

        // new weight = weight + LR * OutError * output(neuron i (before weight))* sigDeriv(neuron i+1 (after weight))

        float[] result = predict(input.clone());
        float[] error = subtract(result,solution);
        float errorSingle = sum(error)/error.length;
        //float errorSingle = error[0];
        float errorSum = (float)((1/error.length)*Math.pow(sum(error),2));
        System.out.println("Error: "+errorSingle);
        float[] derivSigmoidOutput = derivSigmoid(new float[]{solution});
        float[] temp = hidden.clone();
        float[] derivSigmoidNeurons = derivSigmoid(temp);


        // weights HO

        for (int hiddenNeuron = 0; hiddenNeuron < weightsHO.length; hiddenNeuron++) {
            for(int weight = 0; weight < weightsHO[0].length; weight++){
                weightsHO[hiddenNeuron][weight] += learningRate * errorSingle * hidden[hiddenNeuron] * derivSigmoidOutput[weight];
            }
        }

        for (int i = 0; i < biasOutput.length; i++) {
            biasOutput[i] += learningRate * errorSingle  * derivSigmoidOutput[i];
        }

        // weights IH

        for (int hiddenNeuron = 0; hiddenNeuron < weightsIH.length; hiddenNeuron++) {
            for(int weight = 0; weight < weightsIH[0].length; weight++){
                weightsIH[hiddenNeuron][weight] += learningRate * errorSingle * input[weight] * derivSigmoidNeurons[weight];
            }
        }

        for (int i = 0; i < biasHidden.length; i++) {
            biasHidden[i] += learningRate * errorSingle * derivSigmoidNeurons[i];
        }



       return errorSum;

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

    private static float sum(float[] arg1){
        float temp = 0;
        for(int i = 0; i < arg1.length; i++){
            temp += arg1[i];
        }
        return temp;
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

    private static float[] subtract(float[] arg1, float arg2){


        float[] temp = new float[arg1.length];
        for(int i = 0; i < arg1.length; i++){
            temp[i] = arg2-arg1[i];
        }
        return temp;
    }

    private static float[] add(float[] arg1, float[] arg2){

        if(arg1.length != arg2.length){
            System.err.println("Addition error. Arrays are not of equal length!");
        }

        float[] temp = new float[arg1.length];
        for(int i = 0; i < arg1.length; i++){
            temp[i] = arg1[i]+arg2[i];
        }
        return temp;
    }

    private static float[][] add(float[][] arg1, float[] arg2){

        float[][] temp = new float[arg1.length][arg1[0].length];
        for(int i = 0; i < arg1.length; i++){
            temp[i] = add(arg1[i],arg2);
        }
        return temp;
    }

    private static float[][] transpose(float[] arg1){
        float[][] temp = new float[0][arg1.length];

        for(int i = 0; i < arg1.length; i++){
            temp[0][i] = arg1[i];
        }
        return temp;
    }

    private static float[][] transpose(float[][] arg1){
        float[][] temp = new float[arg1[0].length][arg1.length];

        for(int row = 0; row < arg1.length; row++){
            for(int col = 0; col < arg1[0].length; col++){
                temp[col][row] = arg1[row][col];
            }
        }
        return temp;
    }

    private static float multiply(float[] arg1, float[] arg2){

        if(arg1.length != arg2.length){
            System.err.println("Multiplication error. Arrays are not of equal length!");
        }

        float temp = 0;

        for (int i = 0; i < arg1.length; i++) {
            temp += arg1[i]*arg2[i];
        }

        return temp;
    }

    private static float[] multiply(float[] arg1, float[][] arg2){

        if(arg1.length != arg2[0].length){
            System.err.println("Multiplication error. Rows and cols dont match!");
        }

        float[] temp = new float[arg2[0].length];
        for (int i = 0; i < arg1.length; i++) {
            for (int j = 0; j < arg2.length; j++) {
                temp[i] += arg2[j][i] * arg1[j];
            }

        }
        return temp;
    }

    private static float[] multiply(float[] arg1, float arg2){

        float[] temp = new float[arg1.length];
        for (int i = 0; i < arg1.length; i++) {
            temp[i] = arg1[i]*arg2;
        }
        return temp;
    }

    private static float[][] multiply(float[][] arg1, float arg2){

        float[][] temp = new float[arg1.length][arg1[0].length];
        for (int row = 0; row < arg1.length; row++) {
            for (int col = 0; col < arg1[0].length; col++) {
                temp[row][col] = arg1[row][col]*arg2;
            }
        }
        return temp;
    }

    public static void main(String[] args) {

        ANN ann = new ANN(2,3,1);
        float[][] trainData = {{0,0},{1,0},{0,1},{1,1}};
        float[] trainSolutions = {0,1,1,0};

        ann.train(trainData,trainSolutions,0.001f,50000);

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
