import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.FPSAnimator;
import org.joml.Vector2f;

import java.awt.*;
import java.util.LinkedList;

public class Display implements GLEventListener {

    private static final float NODEOFFSET = 65.0f;
    private static final float LAYEROFFSET = 225.0f;
    private static final float SCALE = 25.0f;


    private int width;
    private int height;

    Frame frame;
    GLCanvas canvas;
    GLProfile profile;
    GLCapabilities caps;
    FPSAnimator animator;


    private LinkedList<DisplayNode> nodes = new LinkedList<>();
    private LinkedList<DisplayWeight> weights = new LinkedList<>();

    public Display(int width, int height){

        this.width = width;
        this.height = height;
        createWindow();

    }

    private void createWindow(){

        profile = GLProfile.get(GLProfile.GL2);
        caps = new GLCapabilities(profile);
        canvas = new GLCanvas(caps);
        canvas.addGLEventListener(this);
        canvas.setSize(this.width,this.height);

        frame = new Frame("ANN");
        frame.add(canvas);
        frame.setSize(this.width,this.height);
        frame.setVisible(true);
        animator = new FPSAnimator(canvas, 30, true);
        animator.start();


    }

    public void displayNet(ANN net){

        synchronized (this.nodes){
            this.nodes.clear();
            for (int node = 0; node < net.input.length; node++) {
                this.nodes.add(new DisplayNode(new Vector2f(0,node*NODEOFFSET),net.input[node]));
            }
            for (int hiddenLayer = 0; hiddenLayer < net.hidden.length; hiddenLayer++) {
                for (int node = 0; node < net.hidden[hiddenLayer].length; node++) {
                    this.nodes.add(new DisplayNode(new Vector2f((hiddenLayer+1)*LAYEROFFSET,node*NODEOFFSET),net.hidden[hiddenLayer][node]));
                }
            }
            for (int node = 0; node < net.output.length; node++) {
                this.nodes.add(new DisplayNode(new Vector2f((net.hidden.length+1)*LAYEROFFSET,node*NODEOFFSET),net.output[node]));
            }

        }

        synchronized (this.weights){
            this.weights.clear();
            for(int weightLayer = 0; weightLayer < net.weights.length; weightLayer++){
                for(int weightFrom = 0; weightFrom < net.weights[weightLayer].length; weightFrom++){
                    for(int weightTo = 0; weightTo < net.weights[weightLayer][weightFrom].length; weightTo++){
                        this.weights.add(new DisplayWeight(
                                new Vector2f(weightLayer*LAYEROFFSET,weightFrom*NODEOFFSET),
                                new Vector2f((weightLayer+1)*LAYEROFFSET,weightTo*NODEOFFSET),
                                net.weights[weightLayer][weightFrom][weightTo])
                        );
                    }
                }
            }
        }


    }


    @Override
    public void init(GLAutoDrawable glAutoDrawable) {
        GL2 gl = glAutoDrawable.getGL().getGL2();
        gl.glClearColor(0.1f, 0.1f, 0.1f, 1);
    }

    @Override
    public void dispose(GLAutoDrawable glAutoDrawable) {

    }

    @Override
    public void display(GLAutoDrawable glAutoDrawable) {

        GL2 gl = glAutoDrawable.getGL().getGL2();

        synchronized(this.weights){

            gl.glClear(GL.GL_COLOR_BUFFER_BIT);
            gl.glClear(GL.GL_DEPTH_BUFFER_BIT);

            for(DisplayWeight weight : this.weights){

                float[] vertexPointStart = new float[]{weight.startPosition.x,weight.startPosition.y,0};
                float[] vertexPointEnd = new float[]{weight.endPosition.x,weight.endPosition.y,0};

                Transform.translate(vertexPointStart, new Vector2f(this.width, this.height));
                Transform.translate(vertexPointEnd, new Vector2f(this.width, this.height));

                Transform.translate(vertexPointStart, new Vector2f(-2.0f*LAYEROFFSET, -4*NODEOFFSET));
                Transform.translate(vertexPointEnd, new Vector2f(-2.0f*LAYEROFFSET, -4*NODEOFFSET));

                Transform.toClipSpace(vertexPointStart, this.width, this.height);
                Transform.toClipSpace(vertexPointEnd, this.width, this.height);

                gl.glBegin(GL2.GL_LINES);
                    gl.glColor3f(weight.color.x, weight.color.y, weight.color.z);
                    gl.glVertex3f(vertexPointStart[0],vertexPointStart[1],0);
                    gl.glVertex3f(vertexPointEnd[0],vertexPointEnd[1],0);
                gl.glEnd();

            }

        }
        synchronized (this.nodes) {
            for (DisplayNode node : this.nodes) {

                float[] vertexPoints = node.getVertexPoints();

                Transform.scale(vertexPoints, (node.value + 1) * SCALE);
                Transform.translate(vertexPoints, node.position);
                Transform.translate(vertexPoints, new Vector2f(this.width, this.height));
                Transform.translate(vertexPoints, new Vector2f(-2.0f*LAYEROFFSET, -4*NODEOFFSET));


                Transform.toClipSpace(vertexPoints, this.width, this.height);

                gl.glBegin(GL2.GL_POLYGON);
                    gl.glColor3f(node.color.x, node.color.y, node.color.z);
                    for (int i = 0; i < vertexPoints.length / 3; i++) {
                        gl.glVertex3f(vertexPoints[i * 3], vertexPoints[i * 3 + 1], vertexPoints[i * 3 + 2]);
                    }
                gl.glEnd();

            }
        }
    }

    @Override
    public void reshape(GLAutoDrawable glAutoDrawable, int i, int i1, int i2, int i3) {

    }

}
