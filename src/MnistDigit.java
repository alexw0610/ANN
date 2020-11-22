public class MnistDigit {

    public static void main(String[] args) {

        /*
        The location of the unziped binary files from the mnist database. The first argument is the image data the second one the image labels.
         */
        MnistFileReader mnistObject = new MnistFileReader("./resources/train-images.idx3-ubyte","./resources/train-labels.idx1-ubyte");
        ANN network = new ANN(mnistObject.imgSize,20,5,10);

        //network.loadWeightsFromDisk("./ANN2.model");

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
                network.persistWeightsToDisk("./ANN2.model");
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
        Test the trained neural network on a sample from the mnist database.
         */
        int correct = 0;
        int count = 0;
        for (int testcase = 50; testcase < 250; testcase++) {
            MnistFileReader.TrainingSet test = mnistObject.getTrainingSet(testcase);
            float[] result = network.predict(test.imgData);
            System.out.println("Test Case "+testcase+":");
            System.out.println("Is ["+mnistObject.binaryToLabel(test.imgLabel)+"] Predicted ["+getMax(result)+"]");
            if(mnistObject.binaryToLabel(test.imgLabel) == getMax(result)){
                correct++;
            }
            System.out.println("Is % : Predicted %");
            for (int index = 0; index < 10; index++) {
                System.out.println( test.imgLabel[index]+" : "+result[index]);
            }
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
