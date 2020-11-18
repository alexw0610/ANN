import org.joml.Vector2f;
import org.joml.Vector3f;


public class DisplayNode {

    float value;
    Vector3f color = new Vector3f(255.0f,255.0f,255.0f);
    Vector2f position;

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

    public DisplayNode(Vector2f position, float value){

        this.value = value;
        this.position = position;
        setColor();
    }

    private void setColor(){

        if(this.value < 0.5f){
            this.color = new Vector3f(255.0f,0.0f,0.0f);
        }else{
            this.color = new Vector3f(0.0f,255.0f,0.0f);
        }
    }



    public float[] getVertexPoints(){
        return this.vertexPoints.clone();
    }

}
