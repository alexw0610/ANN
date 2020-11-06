
public class MnistDigit {

    public static void main(String[] args) {

        MnistFileReader mnistObject = new MnistFileReader("./resources/train-images.idx3-ubyte","./resources/train-labels.idx1-ubyte");
        ANN network = new ANN(mnistObject.imgSize,mnistObject.imgSize,30,10);





        double timeSum = 0;
        for(int trainPatch = 0; trainPatch < 50; trainPatch++){
            double startTime = System.currentTimeMillis();
            float[][] input = new float[100][mnistObject.imgSize];
            float[][] solutions = new float[100][];
            for (int i = 0; i < 100; i++) {
                MnistFileReader.TrainingSet set = mnistObject.getTrainingSet(i+trainPatch*10);
                input[i] = set.imgData;
                solutions[i] = set.imgLabel;
            }
            network.train(input,solutions,2000,0.01f,1);
            double endTime = System.currentTimeMillis();
            double timeTaken = endTime-startTime;
            timeSum += timeTaken;
            double timeToFinish = ((timeSum/(trainPatch+1))*(50-trainPatch+1));
            int minutesToFinish = (((int)timeToFinish)/1000)/60;
            int secondsToFinish = (((int)timeToFinish)/1000)%60;

            System.out.println("Training cycle ["+(trainPatch+1)+"] finished.    Estimated time remaining: "+minutesToFinish+"m "+secondsToFinish+"s");
        }


        MnistFileReader.TrainingSet test = mnistObject.getTrainingSet(40);
        float[] result = network.predict(test.imgData);
        System.out.println("Test Case 1:");
        System.out.println("Is: "+mnistObject.binaryToLabel(test.imgLabel));
        mnistObject.printImgToConsole(test.imgData);
        System.out.println("Is : Predicted");
        for (int index = 0; index < 10; index++) {
            System.out.println( test.imgLabel[index]+" : "+Math.round(result[index])+" "+result[index]);
        }

        test = mnistObject.getTrainingSet(56);
        result = network.predict(test.imgData);
        System.out.println("Test Case 2:");
        System.out.println("Is: "+mnistObject.binaryToLabel(test.imgLabel));
        mnistObject.printImgToConsole(test.imgData);
        System.out.println("Is : Predicted");
        for (int index = 0; index < 10; index++) {
            System.out.println( test.imgLabel[index]+" : "+Math.round(result[index])+" "+result[index]);
        }

        test = mnistObject.getTrainingSet(32);
        result = network.predict(test.imgData);
        System.out.println("Test Case 3:");
        System.out.println("Is: "+mnistObject.binaryToLabel(test.imgLabel));
        mnistObject.printImgToConsole(test.imgData);
        System.out.println("Is : Predicted");
        for (int index = 0; index < 10; index++) {
            System.out.println( test.imgLabel[index]+" : "+Math.round(result[index])+" "+result[index]);
        }

        test = mnistObject.getTrainingSet(17);
        result = network.predict(test.imgData);
        System.out.println("Test Case 4:");
        System.out.println("Is: "+mnistObject.binaryToLabel(test.imgLabel));
        mnistObject.printImgToConsole(test.imgData);
        System.out.println("Is : Predicted");
        for (int index = 0; index < 10; index++) {
            System.out.println( test.imgLabel[index]+" : "+Math.round(result[index])+" "+result[index]);
        }



    }

}
