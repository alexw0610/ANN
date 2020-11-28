public class MnistDigit {

    public static void main(String[] args) {

        /*
        The location of the unziped binary files from the mnist database. The first argument is the image data the second one the image labels.
         */
        MnistFileReader mnistObject = new MnistFileReader("./resources/train-images.idx3-ubyte","./resources/train-labels.idx1-ubyte");
        MnistFileReader mnistTestObject = new MnistFileReader("./resources/test-images.idx3-ubyte","./resources/test-labels.idx1-ubyte");

        ANN network = new ANN(mnistObject.imgSize,20,5,10);

        //network.loadWeightsFromDisk("./ANN.model");

        double timeSum = 0;
        for(int trainPatch = 0; trainPatch < 600; trainPatch++){

            double startTime = System.currentTimeMillis();

            float[][] input = new float[100][mnistObject.imgSize];
            float[][] solutions = new float[100][];
            /*
            Pick the first 100 samples from the mnist database.
            Note that the amount of samples you chose here should be somewhat fitting for the epoch amount.
             */
            for (int i = 0; i < 100; i++) {
                MnistFileReader.TrainingSet set = mnistObject.getTrainingSet(i+trainPatch*100);
                input[i] = set.imgData;
                solutions[i] = set.imgLabel;
            }


            network.train(input,solutions,1000,0.01f,1);
            /*
            if (trainPatch%5==0){
                network.persistWeightsToDisk("./ANN.model");
            }
            */
            double endTime = System.currentTimeMillis();
            double timeTaken = endTime-startTime;
            timeSum += timeTaken;
            double timeToFinish = ((timeSum/(trainPatch+1))*(600-trainPatch+1));
            int hoursToFinish = (((int)timeToFinish)/1000)/60/60;
            int minutesToFinish = ((((int)timeToFinish)/1000)/60)%60;
            int secondsToFinish = (((int)timeToFinish)/1000)%60;

            System.out.println("Training cycle ["+(trainPatch+1)+"] finished.\t Estimated time remaining: "+hoursToFinish+"h "+minutesToFinish+"m "+secondsToFinish+"s");
        }



        /*
        Test the trained neural network on a data from the mnist test database.
        For more verbose output during testing remove comment symbols.
         */
        int correct = 0;
        int count = 0;
        for (int testcase = 0; testcase < 10000; testcase++) {
            MnistFileReader.TrainingSet test = mnistTestObject.getTrainingSet(testcase);
            float[] result = network.predict(test.imgData);
            //System.out.println("Test Case "+testcase+":");
            //System.out.println("Is ["+mnistTestObject.binaryToLabel(test.imgLabel)+"] Predicted ["+getMax(result)+"]");
            if(MnistFileReader.binaryToLabel(test.imgLabel) == getMax(result)){
                correct++;
            }
            //System.out.println("Is % : Predicted %");
            //for (int index = 0; index < 10; index++) {
            //    System.out.println( test.imgLabel[index]+" : "+result[index]);
            //}
            count++;
        }

        System.out.println("Prediction accuracy: "+(((float)correct/(float)count)*100)+"%");

    }

    private static int getMax(float[] arg){
        int maxIndex = 0;
        for(int i = 0; i < arg.length;i++){
            maxIndex = arg[i] > arg[maxIndex] ? i : maxIndex;
        }
        return maxIndex;
    }

}
