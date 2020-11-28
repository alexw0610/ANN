import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector4f;


public class DisplayNode {

    float value;
    Vector4f color = new Vector4f(1.0f,1.0f,1.0f,1.0f);
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

        float colorRange = this.value *2 -1;

        if (this.value < 0.5f) {
            this.color = new Vector4f(-colorRange, 0.0f, 0.0f,1);
        } else {
            this.color = new Vector4f(0.0f, colorRange, 0.0f,1);
        }
    }



    public float[] getVertexPoints(){
        return this.vertexPoints.clone();
    }

}
