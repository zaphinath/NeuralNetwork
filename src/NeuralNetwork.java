import java.util.ArrayList;

/**
 * 
 */

/**
 * @author zaphinath
 *
 */
public class NeuralNetwork {
	
	//Need to be set
    private int maxEpochs = 4000;
    private double learnRate = 0.01;
    private int numInput; //Number of input nodes
	private int numHidden; //Number of hidden node layers
	private int numOutput; //Number of output Nodes
	      
	private double[] inputs;
	 
	private double[][] ihWeights; // input-hidden
	private double[] hBiases;
	private double[] hOutputs;
	 
	private double[][] hoWeights; // hidden-output
	private double[] oBiases;
	 
	private double[] outputs;
	 
	// back-prop specific arrays (these could be local to method UpdateWeights)
	private double[] oGrads; // output gradients for back-propagation
	private double[] hGrads; // hidden gradients for back-propagation
	 
	// back-prop momentum specific arrays (these could be local to method Train)
	//private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
	private double[] hPrevBiasesDelta;
	private double[][] hoPrevWeightsDelta;
	private double[] oPrevBiasesDelta;
	
	/**
	 * Needs to add one to every layer except output for a bias
	 * @param numInputs
	 * @param numHiddenLayers
	 * @param numHiddenNodesPerLayer
	 * @param numberOutput
	 */
	public NeuralNetwork(int numInput, int numHidden, int numOutput) {
		this.numInput = numInput; 
		this.numHidden = numHidden; 
		this.numOutput = numOutput; 
		
		this.inputs = new double[numInput];
		
		this.ihWeights = makeMatrix(numInput, numHidden);
		this.hBiases = new double[numHidden];
		this.hOutputs = new double[numHidden];
		
		this.hoWeights = makeMatrix(numHidden, numOutput);
		this.oBiases = new double[numOutput];
		
		this.outputs = new double[numOutput];
		
		// back-prop related arrays below
		this.hGrads = new double[numHidden];
		this.oGrads = new double[numOutput];
		
		//this.ihPrevWeightsDelta = makeMatrix(numInput, numHidden);
		this.hPrevBiasesDelta = new double[numHidden];
		this.hoPrevWeightsDelta = makeMatrix(numHidden, numOutput);
		this.oPrevBiasesDelta = new double[numOutput];
		
	}

	public void train() {
		
	}
	
	public void test() {
		
	}
	
	private double[][] makeMatrix(int rows, int cols) {
		double[][] result = new double[rows][];
        for (int r = 0; r < result.length; ++r) {        	
        	result[r] = new double[cols];
        }
        return result;
	}
	
	private void initWeights() {
		int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        double[] initWeights = new double[numWeights];
        double lo = -0.01;
        double hi = 0.01;
        for (int i=0; i < initWeights.length ; ++i)
                initWeights[i] = (hi - lo) * rnd.nextDouble() + lo;
        this.setWeights(initWeights);
	}
}
