/**
 * 
 */

/**
 * @author Derek Carr
 *
 */
public class main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
//		
//		NeuralNetwork nn = new NeuralNetwork(2,1,new int[]{1},1);
//		//nn.printMatrix(nn.getWeights());
//		//nn.printMatrix(nn.getSigNodes());
//		nn.train(new double[]{1,0}, new double[]{1,0});
//		System.out.println("SigNodes");
//		nn.printMatrix(nn.getSigNodes());
//		System.out.println("DeltaSigNodes");
//		nn.printMatrix(nn.getDeltaSigNodes());
//		System.out.println("DeltaWeights");
//		nn.printMatrix(nn.getDeltaWeights());
//		System.out.println("Weights");
//		nn.printMatrix(nn.getWeights());
		
		NeuralNetwork n2 = new NeuralNetwork(3,1,new int[]{1},2);
		//nn.printMatrix(nn.getWeights());
		//n2.printMatrix(n2.getSigNodes());
		
//		n2.train(new double[]{1,0,1}, new double[]{0,1});
//		System.out.println("Sig Nodes");
//		n2.printMatrix(n2.getSigNodes());
////		System.out.println("Weights");
////		n2.printMatrix(n2.getWeights());
//		System.out.println("DELTA SIG NODES");
//		n2.printMatrix(n2.getDeltaSigNodes());
//		System.out.println("DELTA WEIGHTS");
//		n2.printMatrix(n2.getDeltaWeights());
//		System.out.println("Weights");
//		n2.printMatrix(n2.getWeights());
		
		double[][] iris = IrisData.getSmallRawData();
		NeuralNetwork id = new NeuralNetwork(6,4,new int[]{4,1,1,1}, 2);
		//id.printMatrix(id.getSigNodes());
		for (int i = 0; i < iris.length; i++) {
			double[] green = IrisData.getSmallDataResult();
			if (green[i] == 1) {				
				id.train(iris[i], new double[]{1,0});
			} else {
				id.train(iris[i], new double[]{0,1});				
			}
		}
		id.printMatrix(id.getWeights());
	}

}
