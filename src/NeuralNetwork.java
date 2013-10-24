import java.util.ArrayList;

/**
 * 
 */

/**
 * @author zaphinath
 *
 */
public class NeuralNetwork {
	
	
	//Size Weight Matrix [
	
	
	
	private int maxEpochs = 4000;
	private double learnRate = 0.01;
	private double startWeight = .01;
	
	//Need to be set
	private int numInput; //Number of input nodes
	private int numHidden; //Number of hidden node layers
	private int numOutput; //Number of output Nodes
	private int[] numNodesPerHidden; //Number of nodes in each layer. Each index corresponds to each layer - 0 is first hidden layer
	      
	private double[][] weights; //[eachLayer][nodesperlayer]
	private double[][] sigNodes;//[numhidden+1 (for outputlayer)][nodes per layer]

	public NeuralNetwork(int numInput, int numHidden, int[] numNodesPerHidden, int numOutput) {
		assert numNodesPerHidden.length == numHidden;
		
		this.numInput = numInput;
		this.numHidden = numHidden;
		this.numNodesPerHidden = numNodesPerHidden;
		this.numOutput = numOutput;
		
		this.weights = createWeights(numInput, numHidden, numNodesPerHidden, numOutput);
		this.sigNodes = createSigmoidMatrix(numHidden, numNodesPerHidden,numOutput);
		
	}

	public void train(double[] inputValues, int expectedNode) {
		
	}
	
	public void test() {
		
	}
	/**
	 * Creates a 2D Matrix of varying heights that represents the weight between layers
	 * and initializes the starting weights.
	 * @param numInput2
	 * @param numHidden2
	 * @param numNodesPerHidden2
	 * @param numOutput2
	 * @return
	 */
	private double[][] createWeights(int numInput,
			int numHidden, int[] numNodesPerHidden, int numOutput) {
		double[][] w = new double[numHidden + 1][];
		//Set num weights from input nodes
		w[0] = new double[numInput*numNodesPerHidden[0]];
		for (int i = 0; i < w[0].length; i++) {
			w[0][i] = startWeight;
		}
		//set output of each hidden
		for (int i = 1; i < numHidden; i++) {
			w[i] = new double[numNodesPerHidden[i-1]*numNodesPerHidden[i]];
			for (int j = 0; j < w[i].length; j++) {
				w[i][j] = startWeight;
			}
		}
		//set last weight between last hidden and output
		w[numHidden] = new double[numNodesPerHidden[numHidden-1]*numOutput];
		for (int i = 0; i < w[numHidden].length; i++) {
			w[numHidden][i] = startWeight;
		}		
		return w;
	}

	
	/**
	 * @param numHidden2
	 * @param numNodesPerHidden2
	 * @param numOutput2
	 * @return
	 */
	private double[][] createSigmoidMatrix(int numHidden,
			int[] numNodesPerHidden, int numOutput) {
		double[][] sig = new double[numHidden+1][];
		//create hidden layers
		for (int i = 0; i < sig.length; i++) {
			sig[i] = new double[numNodesPerHidden[i]];
		}
		//create output sigmoid layer
		sig[numHidden+1] = new double[numOutput];
		return sig;
	}

}
