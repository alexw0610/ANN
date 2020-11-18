import org.joml.Vector2f;
import org.joml.Vector3f;

public class DisplayWeight {

    float value;
    Vector3f color = new Vector3f(255.0f, 255.0f, 255.0f);;
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
            this.color = new Vector3f(255.0f, 0.0f, 0.0f);
        } else {
            this.color = new Vector3f(0.0f, 255.0f, 0.0f);
        }

    }


}



