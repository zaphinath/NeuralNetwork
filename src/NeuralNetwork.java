import java.util.ArrayList;

/**
 * 
 */

/**
 * @author zaphinath
 *
 */
public class NeuralNetwork {
	
	private int maxEpochs = 4000;
	private double learnRate = 0.3;
	private double startWeight = .1;
	
	//Need to be set
	private int numInput; //Number of input nodes
	private int numHidden; //Number of hidden node layers
	private int numOutput; //Number of output Nodes
	private int[] numNodesPerHidden; //Number of nodes in each layer. Each index corresponds to each layer - 0 is first hidden layer
	      
	private double[][] weights; //[eachLayer][nodesperlayer]
	private double[][] sigNodes;//[numhidden+1 (for outputlayer)][nodes per layer]
	private double[][] tmpCalcMatrix;//Same size as weights for calculating input*weight before sigmoid is calculated and put in sigNodes

	public NeuralNetwork(int numInput, int numHidden, int[] numNodesPerHidden, int numOutput) {
		assert numNodesPerHidden.length == numHidden;
		
		this.numInput = numInput;
		this.numHidden = numHidden;
		this.numNodesPerHidden = numNodesPerHidden;
		this.numOutput = numOutput;
		
		this.weights = createWeights(numInput, numHidden, numNodesPerHidden, numOutput);
		this.sigNodes = createSigmoidMatrix(numHidden, numNodesPerHidden,numOutput);
		//this.tmpCalcMatrix = cloneEmptyMatrix(weights);
	}


	public void train(double[] inputValues, int expectedOutNode) {
		//calculate sigmoids
		for (int i = 0; i < sigNodes.length; i++) {
			for (int j = 0; j < sigNodes[i].length; j++) {
				// use input values to start with
				double tmpSum = 0;
				if (i == 0) {
					for (int k = 0; k < weights[0].length; k++) {
						int tmpCount = 0; // this is to match the input node position 
						if (k % inputValues.length == j) {
							tmpSum += inputValues[tmpCount]*weights[0][k];
							tmpCount++;
						}
					}
				} else {
					for (int k = 0; k < weights[i].length; k++) {
						int tmpCount = 0; // this is to match the input node position 
						if (k % sigNodes[i].length == j) {
							tmpSum += sigNodes[i-1][tmpCount]*weights[i][k];
							tmpCount++;
						}
					}
				}
				sigNodes[i][j] = 1/(1+Math.pow(Math.E,-(tmpSum)));
			}
		}
	}
	
	public void test(double[] inputValues, int expectedOutNode) {
		
	}
	
	/**
	 * @return the maxEpochs
	 */
	public int getMaxEpochs() {
		return maxEpochs;
	}


	/**
	 * @param maxEpochs the maxEpochs to set
	 */
	public void setMaxEpochs(int maxEpochs) {
		this.maxEpochs = maxEpochs;
	}


	/**
	 * @return the learnRate
	 */
	public double getLearnRate() {
		return learnRate;
	}


	/**
	 * @param learnRate the learnRate to set
	 */
	public void setLearnRate(double learnRate) {
		this.learnRate = learnRate;
	}


	/**
	 * @return the startWeight
	 */
	public double getStartWeight() {
		return startWeight;
	}


	/**
	 * @param startWeight the startWeight to set
	 */
	public void setStartWeight(double startWeight) {
		this.startWeight = startWeight;
	}


	/**
	 * @return the numInput
	 */
	public int getNumInput() {
		return numInput;
	}


	/**
	 * @param numInput the numInput to set
	 */
	public void setNumInput(int numInput) {
		this.numInput = numInput;
	}


	/**
	 * @return the numHidden
	 */
	public int getNumHidden() {
		return numHidden;
	}


	/**
	 * @param numHidden the numHidden to set
	 */
	public void setNumHidden(int numHidden) {
		this.numHidden = numHidden;
	}


	/**
	 * @return the numOutput
	 */
	public int getNumOutput() {
		return numOutput;
	}


	/**
	 * @param numOutput the numOutput to set
	 */
	public void setNumOutput(int numOutput) {
		this.numOutput = numOutput;
	}


	/**
	 * @return the numNodesPerHidden
	 */
	public int[] getNumNodesPerHidden() {
		return numNodesPerHidden;
	}


	/**
	 * @param numNodesPerHidden the numNodesPerHidden to set
	 */
	public void setNumNodesPerHidden(int[] numNodesPerHidden) {
		this.numNodesPerHidden = numNodesPerHidden;
	}


	/**
	 * @return the weights
	 */
	public double[][] getWeights() {
		return weights;
	}


	/**
	 * @param weights the weights to set
	 */
	public void setWeights(double[][] weights) {
		this.weights = weights;
	}


	/**
	 * @return the sigNodes
	 */
	public double[][] getSigNodes() {
		return sigNodes;
	}


	/**
	 * @param sigNodes the sigNodes to set
	 */
	public void setSigNodes(double[][] sigNodes) {
		this.sigNodes = sigNodes;
	}


	/**
	 * @return the tmpCalcMatrix
	 */
	public double[][] getTmpCalcMatrix() {
		return tmpCalcMatrix;
	}


	/**
	 * @param tmpCalcMatrix the tmpCalcMatrix to set
	 */
	public void setTmpCalcMatrix(double[][] tmpCalcMatrix) {
		this.tmpCalcMatrix = tmpCalcMatrix;
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
		for (int i = 0; i < numHidden; i++) {
			sig[i] = new double[numNodesPerHidden[i]];
		}
		//create output sigmoid layer
		sig[numHidden] = new double[numOutput];
		return sig;
	}
	
//	/**
//	 * @param weights
//	 * @return
//	 */
//	private double[][] cloneEmptyMatrix(double[][] weights) {
//		double[][] clone = new double[weights.length][];
//		for (int i = 0; i < weights[0].length; i++) {
//			clone[i] = new double[weights[i].length];
//		}
//		return null;
//	}

	public void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.println("i=" + i + "  j=" + j + "  value="+matrix[i][j]);
			}
		}
	}
}
