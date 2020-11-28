import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector4f;

public class DisplayWeight {

    float value;
    Vector4f color = new Vector4f(1.0f, 1.0f, 1.0f,1.0f);;
    Vector2f startPosition;
    Vector2f endPosition;


    public DisplayWeight(Vector2f start, Vector2f end, float value) {

        this.value = value;
        this.startPosition = start;
        this.endPosition = end;
        setColor();
    }

    private void setColor() {

        if (this.value < 0f) {
            this.color = new Vector4f(-this.value, 0.0f, 0.0f,Math.abs(this.value));
        } else {
            this.color = new Vector4f(0.0f, this.value, 0.0f,Math.abs(this.value));
        }

    }


}



