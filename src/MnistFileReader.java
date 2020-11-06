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

    public void printImgToConsole(float[] data){
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

    private float[] labelToBinary(int label){

        float[] temp = new float[10];
        for(int i = 0; i < temp.length; i++) {
            temp[i] = i==label?1:0;
        }
        return temp;

    }
    public int binaryToLabel(float[] binaryLabel){
        for (int i = 0; i < binaryLabel.length; i++) {
            if(binaryLabel[i] == 1){
                return i;
            }
        }
        return 0;
    }

    public class TrainingSet{

        public float[] imgData;
        public float[] imgLabel;
        public TrainingSet(float[] imgData, float[] imgLabel){
            this.imgData = imgData;
            this.imgLabel = imgLabel;
        }

    }

}
