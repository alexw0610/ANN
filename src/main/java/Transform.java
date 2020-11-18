import org.joml.Vector2f;

public class Transform {

    public static float[] toClipSpace(float[] vertexPoints, int width, int height){

            for (int i = 0; i <vertexPoints.length ; i++) {
                if((i%3)==0){
                    vertexPoints[i] = ((vertexPoints[i]-(width))/width);
                }else if((i%3)==1){
                    vertexPoints[i] = ((vertexPoints[i]-(height))/height);
                }
            }

        return vertexPoints;
    }

    public static float[] scale(float[] vertexPoints, float factor){

        for (int i = 0; i <vertexPoints.length ; i++) {
            vertexPoints[i] *= factor;
        }
        return vertexPoints;

    }

    public static float[] translate(float[] vertexPoints, Vector2f position){

        for (int i = 0; i <vertexPoints.length ; i++) {
            if((i%3)==0){
                vertexPoints[i] += position.x;
            }else if((i%3)==1){
                vertexPoints[i] += position.y;
            }
        }
        return vertexPoints;

    }


}
