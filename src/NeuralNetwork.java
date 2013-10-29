import java.util.ArrayList;
import java.util.Random;

/**
 * 
 */

/**
 * @author zaphinath
 *
 */
public class NeuralNetwork extends SupervisedLearner {
	
	private int maxEpochs = 1000;
	private double learnRate = 0.5;
	private double startWeightMax = .2;
	private double startWeightMin = -.2;
	private double momentum = .5;
	private boolean haveMomentum = false;
	private double stoppingThreshold = .05;
	
	//Need to be set
	private int numInput; //Number of input nodes
	private int numHidden; //Number of hidden node layers
	private int numOutput; //Number of output Nodes
	private int[] numNodesPerHidden; //Number of nodes in each layer. Each index corresponds to each layer - 0 is first hidden layer
	      
	private double[][] weights; //[eachLayer][nodesperlayer]
	private double[][] sigNodes;//[numhidden+1 (for outputlayer)][nodes per layer]
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


	/**
	 * @param rand
	 */
	public NeuralNetwork(Random rand) {
		// TODO Auto-generated constructor stub
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
						//TODO possible bug because i think there is a 3rd case here
						preSigmoid[k] = weights[i][k]*inputValues[tmpCount];
						if (weights[i].length > inputValues.length) {
							if (k%inputValues.length == 0) {
								tmpCount++;
							}
						} else {
							tmpCount=k;
						}
						if (tmpCount >= inputValues.length) {
							break;
						}
						//System.out.println(weights[i][k]);
						//System.out.println(inputValues[tmpCount]);
						//System.out.println("tmpCount = "+tmpCount + "  k = "+k);
						
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
				sigNodes[i][j] = 1/(1+Math.pow(Math.E,-(tmpSum)));
			}
		}
		//printMatrix(sigNodes);
		//Sigmoids are calculated 
		//BackPropogate and update weights
		// calculate deltasigmoids && deltaweights
		for (int i = deltaSigNodes.length-1; i >= 0; i--) {
			for (int j = 0; j < deltaSigNodes[i].length; j++) {
				//Out nodes have diff formula
				if (i == deltaSigNodes.length-1) {
					deltaSigNodes[i][j] = sigNodes[i][j]*(1-sigNodes[i][j])*(expectedOutValues[j]-sigNodes[i][j]);
				//Hidden Nodes
				} else {
					if ( deltaSigNodes[i].length < weights[i+1].length) {
						// two more cases when dsig < weights+1
						// Need to loop through weights
						//TODO - formula may be wrong - may want weights*deltaweights
						double tmpSum = 0;
						int tmpCount2 = 0; // keeps track of deltasig+1 col
						//TODO Potential problem in calculation - watch carefully
						//We want to start counting at the weights array at a new point based off the signodes before it's position
						for (int k = (j/deltaSigNodes[i].length)*weights[i+1].length; k < weights[i+1].length-1; k++) {
							if (weights[i+1].length == deltaSigNodes[i+1].length) {
								tmpSum += (weights[i+1][k] * deltaSigNodes[i+1][k]);
							} else {
								// dsig+1.length < weights+1.length
								//uses tmpcount2
								tmpSum += weights[i+1][tmpCount2] * deltaSigNodes[i+1][k];
								if (weights[i+1].length % sigNodes[i+1].length == 0) {
									break;
								}
								tmpCount2++;
							}
						}
						//System.out.println("tmpSum:"+tmpSum);
						deltaSigNodes[i][j] = sigNodes[i][j]*(1-sigNodes[i][j])*(tmpSum);
					} else {
						// case where they are equal. it will never be less
						deltaSigNodes[i][j] = sigNodes[i][j]*(1-sigNodes[i][j])*(deltaWeights[i+1][j]*weights[i+1][j]);
					}
				}
			}
			//update delta weights @ i
			int tmpCount = 0; //to delta[i]
			int tmpCount2 = 0; // to signode-1
			for (int k = 0; k < deltaWeights[i].length; k++) {
				//Need to account for end of array
				if (i > 0) {
					deltaWeights[i][k] = learnRate * deltaSigNodes[i][tmpCount] * sigNodes[i-1][tmpCount2];
					//for tmpcount
					if (deltaSigNodes[i].length > deltaWeights[i].length) {
						// won't happen according to my design
					} else if (deltaSigNodes[i].length == deltaWeights[i].length ){
						tmpCount = k;
					} else {
						// case where deltaSigNodes[i].length < deltaWeights[i].length
						//TODO
						//System.out.println("WARNING");
						if (k/deltaSigNodes[i].length == 1) {
							tmpCount++;
//							System.out.println(tmpCount);
//							System.out.println("LENGTH: "+ deltaSigNodes[i].length);
							if (tmpCount == deltaSigNodes[i].length) {
								tmpCount--;
							}
						}
						
					}
					//for tmpcount2
					if (deltaSigNodes[i-1].length > deltaWeights[i].length) {
						//don't think this case can ever happen
						//multiple nodes to count because dnodes is larger
					} else if (deltaSigNodes[i-1].length == deltaWeights[i].length) {
						tmpCount2 = k;
					} else {
						if (k%deltaWeights[i].length == 0) {
							tmpCount2++;
						}
					}
					
				} else {
					//System.out.println("tmpCount2=" + tmpCount2 + "  inputLength="+inputValues.length + "  deltaWeights[i].length="+deltaWeights[i].length + "  k="+k);
					deltaWeights[i][k] = learnRate * deltaSigNodes[i][tmpCount] * inputValues[tmpCount2];
					//Where i = 0 and we need to use input nodes.
					if (inputValues.length == deltaWeights[i].length) {
						tmpCount2 = k;
					} else {
						if (k/deltaSigNodes[i].length == 1) {
							tmpCount2++;
						}
					}

				}
				
			}
		}
		//update weights
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				weights[i][j] += deltaWeights[i][j];
			}
		}
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
	 * @return the startWeightMax
	 */
	public double getStartWeightMax() {
		return startWeightMax;
	}


	/**
	 * @param startWeightMax the startWeightMax to set
	 */
	public void setStartWeightMax(double startWeightMax) {
		this.startWeightMax = startWeightMax;
	}


	/**
	 * @return the startWeightMin
	 */
	public double getStartWeightMin() {
		return startWeightMin;
	}


	/**
	 * @param startWeightMin the startWeightMin to set
	 */
	public void setStartWeightMin(double startWeightMin) {
		this.startWeightMin = startWeightMin;
	}


	/**
	 * @return the stoppingThreshold
	 */
	public double getStoppingThreshold() {
		return stoppingThreshold;
	}


	/**
	 * @param stoppingThreshold the stoppingThreshold to set
	 */
	public void setStoppingThreshold(double stoppingThreshold) {
		this.stoppingThreshold = stoppingThreshold;
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
			double random = new Random().nextDouble();
			double result = startWeightMin + (random * (startWeightMax - startWeightMin));
			w[0][i] = .2;
		}
		//set output of each hidden
		for (int i = 1; i < numHidden; i++) {
			w[i] = new double[numNodesPerHidden[i-1]*numNodesPerHidden[i]];
			for (int j = 0; j < w[i].length; j++) {
				double random = new Random().nextDouble();
				double result = startWeightMin + (random * (startWeightMax - startWeightMin));
				w[i][j] = .2;
			}
		}
		//set last weight between last hidden and output
		w[numHidden] = new double[numNodesPerHidden[numHidden-1]*numOutput];
		for (int i = 0; i < w[numHidden].length; i++) {
			double random = new Random().nextDouble();
			double result = startWeightMin + (random * (startWeightMax - startWeightMin));
			w[numHidden][i] = .2;
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
			System.out.print("i=" + i + "  (");
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.print("["+matrix[i][j]+"] ");
			}
			System.out.println(")");
		}
	}

	/**
	 * @return the momentum
	 */
	public double getMomentum() {
		return momentum;
	}

	/**
	 * @param momentum the momentum to set
	 */
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	/**
	 * @return the haveMomentum
	 */
	public boolean isHaveMomentum() {
		return haveMomentum;
	}

	/**
	 * @param haveMomentum the haveMomentum to set
	 */
	public void setHaveMomentum(boolean haveMomentum) {
		this.haveMomentum = haveMomentum;
	}

	private double[] getResultsArray(Matrix labels, int index) {
		double[] result = new double[labels.valueCount(0)];
		
		for (int i = 0; i < result.length; i++) {
			//System.out.println("labels=" + (int)labels.get(i, 0) + "  i=" + i);
			if (labels.get(index, 0) == i) {
				result[i] = 1;
				//System.out.println("HERE");
			} else {
				result[i] = 0;
			}
		}
//		System.out.print("RESULT[");
//		for (int i = 0; i < result.length; i++) {
//			System.out.print("i=" + i + "  [i]="+ result[i] + "  ");
//		}
//		System.out.println("] @ index="+index);
		return result;
	}
	
	/* (non-Javadoc)
	 * @see SupervisedLearner#train(Matrix, Matrix)
	 */
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		this.numInput = features.cols();
		this.numHidden = 3;
		this.numNodesPerHidden = new int[]{3,3,3};
		this.numOutput = labels.valueCount(0);
		
		this.weights = createWeights(numInput, numHidden, numNodesPerHidden, numOutput);
		this.sigNodes = createSigmoidMatrix(numHidden, numNodesPerHidden,numOutput);
		this.deltaSigNodes = cloneEmptyMatrix(sigNodes);
		this.deltaWeights = cloneEmptyMatrix(weights);
		
		//printMatrix(weights);
		
		//loop through all epocs and check for breaking conditions

		for (int h = 0; h < maxEpochs; h++) {
			//printMatrix(deltaWeights);
			for (int i = 0; i< features.rows(); i++) {
				double [] end = getResultsArray(labels, i);
				//System.out.println("Lables" + labels.get(i, 0));
				train(features.row(i), end);
//				for (int foo = 0; foo < features.row(i).length; foo++) {
//					System.out.print(features.row(i)[foo]+ "  ");
//					System.out.println("result="+labels.get(i, 0));
//				}
			}
			//printMatrix(weights);
			// check breaking conditions met
			//printMatrix(sigNodes);
			boolean thresholdMet = true;
			for (int i = 0; i < deltaWeights.length; i++) {
				for (int j = 0; j < deltaWeights[i].length; j++) {
					if (deltaWeights[i][j] > stoppingThreshold) {
						thresholdMet = false;
						break;
					}
					if (!thresholdMet) {
						break;
					}
				}
				if (!thresholdMet){
					break;
				}
			}
			if (!thresholdMet) {
				break;
			}
		}
		//printMatrix(deltaSigNodes);
		//printMatrix(weights);
	}


	/* (non-Javadoc)
	 * @see SupervisedLearner#predict(double[], double[])
	 */
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		//First we take each node times a weight and store it in deltaweights for tmp
		//Next we add the deltaweight tmp storage to the right sigNode
		//Then we determine the output node which is the highest and return that value
		for (int i = 0; i < sigNodes.length; i++) {
			int tmpCount = 0; // use to keep track of input nodes
			//update deltaWeights with tmp calc for each i
			for (int j = 0; j < deltaWeights[i].length; j++) {
				// uses input nodes
				if (i == 0) {
					deltaWeights[i][j] = weights[i][j]*features[tmpCount];
					if (j % features.length == 0) {
						tmpCount++;
					}
				} else {
					deltaWeights[i][j] = weights[i][j]*sigNodes[i-1][tmpCount];
					if (deltaWeights[i].length == sigNodes[i-1].length) {
						tmpCount++;
					} else {
						if (j % sigNodes[i-1].length == 0) {
							tmpCount++;
							if (tmpCount >= sigNodes[i-1].length) {
								tmpCount = sigNodes[i-1].length -1;
							}
						}
					}
				}
			}
			//update signode at each i
			tmpCount = 0;
			for (int j = 0; j < deltaWeights[i].length; j++) {
				sigNodes[i][tmpCount] += deltaWeights[i][j];
				if (j % sigNodes[i].length == 0) {
					tmpCount = 0;
				} else {
					tmpCount++;
				}
			}
		}
		//printMatrix(sigNodes);
		labels[0] = 0;
		//System.out.println("Signodes Length: " + sigNodes.length);
		for (int i = 1; i < numOutput; i++) {
			//System.out.println("i=" + i +"  @i="+sigNodes[sigNodes.length-1][i] + "  @i-1="+sigNodes[sigNodes.length-1][i-1]);
			if (sigNodes[sigNodes.length-1][i] > sigNodes[sigNodes.length-1][i-1] ) {
				labels[0] = i;
			}
		}
		//System.out.println(labels[0]);
	}
}
