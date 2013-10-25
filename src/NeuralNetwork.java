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
	private double learnRate = 0.5;
	private double startWeight = .2;
	
	//Need to be set
	private int numInput; //Number of input nodes
	private int numHidden; //Number of hidden node layers
	private int numOutput; //Number of output Nodes
	private int[] numNodesPerHidden; //Number of nodes in each layer. Each index corresponds to each layer - 0 is first hidden layer
	      
	private double[][] weights; //[eachLayer][nodesperlayer]
	private double[][] sigNodes;//[numhidden+1 (for outputlayer)][nodes per layer]
	//private double[][] tmpCalcMatrix;//Same size as weights for calculating input*weight before sigmoid is calculated and put in sigNodes
	private double[][] deltaSigNodes; // the deltaSigmoid backprop calc
	private double[][] deltaWeights; // the delta in weight changes 
	
	public NeuralNetwork(int numInput, int numHidden, int[] numNodesPerHidden, int numOutput) {
		assert numNodesPerHidden.length == numHidden;
		
		this.numInput = numInput;
		this.numHidden = numHidden;
		this.numNodesPerHidden = numNodesPerHidden;
		this.numOutput = numOutput;
		
		this.weights = createWeights(numInput, numHidden, numNodesPerHidden, numOutput);
		this.sigNodes = createSigmoidMatrix(numHidden, numNodesPerHidden,numOutput);
		//this.tmpCalcMatrix = cloneEmptyMatrix(weights);
		this.deltaSigNodes = cloneEmptyMatrix(sigNodes);
		this.deltaWeights = cloneEmptyMatrix(weights);
	}


	public void train(double[] inputValues, double[] expectedOutValues) {
		//calculate sigmoids
		for (int i = 0; i < sigNodes.length; i++) {
			for (int j = 0; j < sigNodes[i].length; j++) {
				double tmpSum = 0;
				// first sigNode array - uses input values
				if (i == 0) {
					// Make tmp array of values input * weight => same size as weight
					double[] preSigmoid = new double[weights[i].length];
					int tmpCount = 0; // corresponds to input node
					// Loop through weights
					for (int k = 0; k < weights[i].length; k++) {
						if (weights[i].length > inputValues.length) {
							if (k/inputValues.length == 1) {
								tmpCount++;
							}
						} else {
							tmpCount=k;
						}
						preSigmoid[k] = weights[i][k]*inputValues[tmpCount];
//						System.out.println("INPUT[tmpCount]" + inputValues[tmpCount]);
//						System.out.println("Presig[k]=" + preSigmoid[k]);
					}
					//Add tmp values per sigNodes[i][j];
					for ( int k = 0; k < preSigmoid.length; k++) {
						if (k % sigNodes[i].length == j) {
							//System.out.println("K="+ k + " preSig=" + preSigmoid[k]);
							tmpSum += preSigmoid[k];
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
//				System.out.println(tmpSum);
				sigNodes[i][j] = 1/(1+Math.pow(Math.E,-(tmpSum)));
			}
		}
		//Sigmoids are calculated 
		//BackPropogate and update weights
		// calculate deltasigmoids && deltaweights
		for (int i = deltaSigNodes.length-1; i > 0; i--) {
			for (int j = 0; j < deltaSigNodes[i].length; j++) {
				//Out nodes have diff formula
				if (i == deltaSigNodes.length-1) {
					deltaSigNodes[i][j] = sigNodes[i][j]*(1-sigNodes[i][j])*(expectedOutValues[j]-sigNodes[i][j]);
				} else {
					deltaSigNodes[i][j] = sigNodes[i][j]*(1-sigNodes[i][j])*(expectedOutValues[j]-sigNodes[i][j]);
				}
			}
			//update delta weights @ i
//			int tmpCount = 0; //to delta[i]
//			int tmpCount2 = 0; // to signode-1
//			for (int k = 0; k < deltaWeights[i].length; k++) {
//				if (deltaWeights[i].length > deltaSigNodes[i].length) {
//					if (k/deltaWeights[i].length == 1) {
//						tmpCount++;
//					}
//				} else {
//					tmpCount=k;
//				} 
//				if (i >= 1) {
//					if (deltaWeights[i].length > SigNodes[i-1]) {
//						if (k/deltaWeights[i].length == 1) {
//							tmpCount++;
//						}
//					}
//				}
//				deltaWeights[i][k] = learnRate*deltaSigNodes[]
//			}
		}
		
		//update weights
		
	}
	
	public void test(double[] inputValues, double[] expectedOutValues) {
		
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

//	/**
//	 * @return the tmpCalcMatrix
//	 */
//	public double[][] getTmpCalcMatrix() {
//		return tmpCalcMatrix;
//	}
//
//
//	/**
//	 * @param tmpCalcMatrix the tmpCalcMatrix to set
//	 */
//	public void setTmpCalcMatrix(double[][] tmpCalcMatrix) {
//		this.tmpCalcMatrix = tmpCalcMatrix;
//	}


	/**
	 * @return the deltaSigNodes
	 */
	public double[][] getDeltaSigNodes() {
		return deltaSigNodes;
	}


	/**
	 * @param deltaSigNodes the deltaSigNodes to set
	 */
	public void setDeltaSigNodes(double[][] deltaSigNodes) {
		this.deltaSigNodes = deltaSigNodes;
	}


	/**
	 * @return the deltaWeights
	 */
	public double[][] getDeltaWeights() {
		return deltaWeights;
	}


	/**
	 * @param deltaWeights the deltaWeights to set
	 */
	public void setDeltaWeights(double[][] deltaWeights) {
		this.deltaWeights = deltaWeights;
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
	
	/**
	 * @param weights
	 * @return
	 */
	private double[][] cloneEmptyMatrix(double[][] matrix) {
		double[][] clone = new double[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			clone[i] = new double[matrix[i].length];
		}
		return clone;
	}

	public void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.println("i=" + i + "  j=" + j + "  value="+matrix[i][j]);
			}
		}
	}
}
