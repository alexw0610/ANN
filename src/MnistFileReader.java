import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;

public class MnistFileReader {

    ByteBuffer labelValues;
    int labelAmount;

    ByteBuffer imgValues;
    int imgAmount;
    int imgRowDim;
    int imgColDim;
    int imgSize;

    public MnistFileReader(String imgPath, String labelPath){
        loadLabelInfo(labelPath);
        loadImgInfo(imgPath);
    }

    /**
     * load the binary label file into a byte array nad
     * load the amount of labels variable contained in the header of the file.
     * @param path
     */
    private void loadLabelInfo(String path){

        File file = new File(path);
        try {
            byte[] tempBytes = Files.readAllBytes(file.toPath());
            ByteBuffer buffer = ByteBuffer.wrap(tempBytes);
            this.labelAmount = buffer.getInt(4);
            this.labelValues = buffer;

        } catch (IOException e) {
            System.err.println("Error opening file in path:\n"
                    +path+"\n"
                    +"Is the file opened by another process?");
            e.printStackTrace();
        }

    }

    /**
     * load the binary image file into a byte array and
     * set the amount of images and image dimensions variables contained in the header of the file.
     * @param path
     */
    private void loadImgInfo(String path){

        File file = new File(path);
        try {
            byte[] tempBytes = Files.readAllBytes(file.toPath());
            ByteBuffer buffer = ByteBuffer.wrap(tempBytes);
            this.imgAmount = buffer.getInt(4);
            this.imgRowDim = buffer.getInt(8);
            this.imgColDim = buffer.getInt(12);
            this.imgSize = this.imgColDim*this.imgRowDim;
            this.imgValues = buffer;


        } catch (IOException e) {
            System.err.println("Error opening file in path:\n"
                    +path+"\n"
                    +"Is the file opened by another process?");
            e.printStackTrace();
        }
    }


    /**
     * get the image data for a single image from the mnist database
     * @param index the index of the image to be retrieved
     * @return an array contained the pixels of the image (0-255) on a row by row basis.
     */
    public int[] getImage(int index){
        if(index < 0 && index > imgAmount){
            System.err.println("Cant retrieve img #"+index+", only "+imgAmount+" images loaded!");
            System.exit(0);
        }

        int[] temp = new int[this.imgSize];
        for(int pixel = 0; pixel < this.imgSize;pixel++){
            temp[pixel] = imgValues.get(index*this.imgSize+pixel+16);
        }
        return temp;
    }

    /**
     * get the image data for a single image from the mnist database
     * @param index the index of the image set to be retrieved
     * @return An object containing the array of the pixels of the image (0-255) on a row by row basis and the array containing the binary representation of the label
     */
    public TrainingSet getTrainingSet(int index){

        if(index < 0 && index > imgAmount){
            System.err.println("Cant retrieve img #"+index+", only "+imgAmount+" images loaded!");
            System.exit(0);
        }


        float[] tempData = new float[this.imgSize];

        //pack image data into array
        for(int pixel = 0; pixel < this.imgSize;pixel++){
            tempData[pixel] = Byte.toUnsignedInt(imgValues.get(index*this.imgSize+pixel+16))/255.0f;
        }

        //retrieve image label
        int tempLabel = labelValues.get(index+8);


        return new TrainingSet(tempData,labelToBinary(tempLabel));

    }

    /**
     * print the image data from a mnist image to the console
     * @param data the array containing the image data
     */
    public static void printImgToConsole(float[] data){
        System.out.println("\n");
        for (int row = 0; row < this.imgRowDim; row++) {
            for (int col = 0; col < this.imgColDim; col++) {
                if(data[row*this.imgRowDim+col]<0.5f){
                    System.out.print("_");
                }else{
                    System.out.print("#");
                }
            }
            System.out.print("\n");
        }
        System.out.println("\n");

    }

    /**
     * pack the representation of the label into a binary format that coresponds to the output neuron of a neural network
     * e.g: lable = 8 -> 0,0,0,0,0,0,0,0,1,0 the 9th neuron lights up indicating a digit 8 was processed.
     * @param label the int representation of the label [0-9]
     * @return the array containing the binary representation
     */
    public static float[] labelToBinary(int label){

        float[] temp = new float[10];
        for(int i = 0; i < temp.length; i++) {
            temp[i] = i==label?1:0;
        }
        return temp;

    }

    /**
     * same as labelToBinary but in reverse
     * @param binaryLabel the array containing the binary representation
     * @return the integer representation of the label
     */
    public static int binaryToLabel(float[] binaryLabel){
        for (int i = 0; i < binaryLabel.length; i++) {
            if(binaryLabel[i] == 1){
                return i;
            }
        }
        return 0;
    }

    /**
     * An object class used to store the image data together with the label data.
     * Used a return object.
     */
    public class TrainingSet{

        public float[] imgData;
        public float[] imgLabel;
        public TrainingSet(float[] imgData, float[] imgLabel){
            this.imgData = imgData;
            this.imgLabel = imgLabel;
        }

    }

}
