import org.joml.Vector2f;


public class DisplayNode {

    private static final float MODIFIER = 4.0f;
    private static final float LAYERDISTANCE = 100.0f;
    private static final float NODEDISTANCE = 50.0f;
    float value;

    Vector2f position = new Vector2f();


    float[] vertexPoints = new float[]{
            0f,0.5f,0f,
            -0.5f,0.2f,0f,
            -0.5f,-0.2f,0f,
            0f,-0.5f,0f,
            0f,0.5f,0f,
            0.5f,0.2f,0f,
            0.5f,-0.2f,0f,
            0f,-0.5f,0f
    };

    public DisplayNode(int layer,int index,float value){

        this.value = value;
        this.position.x = layer;
        this.position.y = index;

    }

    public void toScreenCoordinates(int width, int height){
        for (int i = 0; i <vertexPoints.length ; i++) {
            if((i%3)==0){
                vertexPoints[i] = vertexPoints[i]/width;
            }else if((i%3)==1){
                vertexPoints[i] = vertexPoints[i]/height;
            }
        }
    }

    public void scaleVertexPoints(){
        for (int i = 0; i <vertexPoints.length ; i++) {
            vertexPoints[i] *= (MODIFIER);
        }
    }

    public void translateVertexPoints(){
        for (int i = 0; i <vertexPoints.length ; i++) {
            if((i%3)==0){
                vertexPoints[i] += this.position.x * LAYERDISTANCE;
            }else if((i%3)==1){
                vertexPoints[i] += this.position.y * NODEDISTANCE;
            }
        }
    }

    public float[] getVertexPoints(){
        return this.vertexPoints;
    }

}
