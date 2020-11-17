
import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.GLCanvas;
import java.awt.*;
import java.nio.FloatBuffer;
import java.util.LinkedList;

public class Display implements GLEventListener {

    private int width;
    private int height;

    Frame frame;
    GLCanvas canvas;
    GLProfile profile;
    GLCapabilities caps;


    private LinkedList<DisplayNode> nodes = new LinkedList<>();

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
        frame.setSize(this.width,this.height);
        canvas.setSize(this.width,this.height);

    }

    public void displayNet(ANN net){
        this.nodes.clear();

        for (int node = 0; node < net.input.length; node++) {
            this.nodes.add(new DisplayNode(0,node,net.input[node]));
        }

        for (int hiddenLayer = 0; hiddenLayer < net.hidden.length; hiddenLayer++) {
            for (int node = 0; node < net.hidden[hiddenLayer].length; node++) {
                this.nodes.add(new DisplayNode(hiddenLayer+1,node,net.hidden[hiddenLayer][node]));
            }
        }

        for (int node = 0; node < net.output.length; node++) {
            this.nodes.add(new DisplayNode(net.hidden.length+1,node,net.output[node]));
        }
        canvas.display();

    }




    @Override
    public void init(GLAutoDrawable glAutoDrawable) {

    }

    @Override
    public void dispose(GLAutoDrawable glAutoDrawable) {

    }

    @Override
    public void display(GLAutoDrawable glAutoDrawable) {

        GL2 gl = glAutoDrawable.getGL().getGL2();

        for (DisplayNode node: this.nodes) {
            node.scaleVertexPoints();
            node.translateVertexPoints();
            node.toScreenCoordinates(this.width,this.height);


            System.out.println(node.vertexPoints[0]);

            gl.glBegin(GL2.GL_POLYGON);
            for (int i = 0; i < node.vertexPoints.length/3; i++) {
                gl.glVertex3f(node.vertexPoints[i*3],node.vertexPoints[i*3+1],node.vertexPoints[i*3+2]);
            }


            gl.glEnd();

        }
        this.nodes.clear();


    }

    @Override
    public void reshape(GLAutoDrawable glAutoDrawable, int i, int i1, int i2, int i3) {

    }

    public static void main(String[] args) {
        Display display = new Display(1024,768);
    }
}
