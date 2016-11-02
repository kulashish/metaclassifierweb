package in.ac.iitb.cse.qh.marti;

import java.util.logging.Level;
import java.util.logging.Logger;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;
import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.data.CurrentState;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.ModelParams;
import in.ac.iitb.cse.qh.data.TargetState;
import in.ac.iitb.cse.qh.meta.ClassifierProxy;
import in.ac.iitb.cse.qh.meta.Optimizer;
import in.ac.iitb.cse.qh.meta.TargetStateCalculator;
import in.ac.iitb.cse.qh.util.MetaConstants;
import in.ac.iitb.cse.qh.util.WekaUtil;
import weka.core.Instance;
import weka.core.Instances;

public class AdaptiveMartiNode {
	private static final Logger LOGGER = Logger
			.getLogger(AdaptiveMartiNode.class.getName());

	private int number = 0;
	private ClassifierProxy cProxy;
	private NodeInputData inData;
	private NodeOutputData outData;
	private ModifiedLogistic classifier;
	private boolean blnTrained;
	private AdaptiveMartiLevel level;

	public AdaptiveMartiNode(Instances trainInstances,
			Instances holdoutInstances, AdaptiveMartiLevel level, int number) {
		inData = new NodeInputData(trainInstances, holdoutInstances);
		outData = new NodeOutputData(trainInstances.numInstances(),
				holdoutInstances.numInstances());
		cProxy = new ClassifierProxy();
		this.level = level;
		this.number = number;
	}

	public void display() {
		LOGGER.log(Level.INFO, "Node number: " + number);
	}

	public int getNumber() {
		return number;
	}

	public void setNumber(int number) {
		this.number = number;
	}

	public AdaptiveMartiLevel getLevel() {
		return level;
	}

	public void setLevel(AdaptiveMartiLevel level) {
		this.level = level;
	}

	public void train() throws Exception {
		if (MetaConstants.BALANCE_TRAINING_DATA)
			WekaUtil.balanceInstances(inData.trainingInstances);
		InputData inputData = cProxy.computeInitialState(
				inData.trainingInstances, inData.holdoutInstances);

		if (MetaConstants.TUNING) {
			if (level.getLevelNumber() == 0 && number == 0) // root node
				inputData.createDefaultBiasLowFP(0);
			else
				inputData
						.createBias(inData.getTargetFP(), inData.getTargetFN());
			CurrentState currState = CurrentState.createCurrentState(inputData
					.getPredInstances());
			TargetStateCalculator tstateCalc = new TargetStateCalculator(
					inputData, currState);
			TargetState targetState = tstateCalc.calculate();
			Optimizer optimizer = new Optimizer(inputData, currState,
					targetState, cProxy);
			ModelParams params = null;
			try {
				params = optimizer.optimize2();
			} catch (Exception e) {
				e.printStackTrace();
				params = null;
			}
			classifier = cProxy.trainModel(params);
		} else
			classifier = cProxy.getClassifier();

		blnTrained = true;
	}

	public void build() throws Exception {
		train();
		outData = new NodeOutputData(inData.trainingInstances.numInstances(),
				inData.holdoutInstances.numInstances());
		classifyRoute(inData.trainingInstances, outData.trainProb,
				outData.trainConfusionMatrix, true);
		classifyRoute(inData.holdoutInstances, outData.holdoutProb,
				outData.holdoutConfusionMatrix, false);
	}

	private void classifyRoute(Instances instances, double[] prob,
			ConfusionMatrix confusionMatrix, boolean blnTrain) throws Exception {
		LOGGER.log(Level.INFO,
				"Classifying instances :" + instances.numInstances());
		int[][] conf = WekaUtil.classify(classifier, instances, prob);
		confusionMatrix = new ConfusionMatrix(conf);
		LOGGER.log(Level.INFO, "Confusion Matrix:");
		confusionMatrix.display();
		Instance instance = null;
		double gamma = 0.0d;
		int nextLevelIndex = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			instance = instances.get(i);
			gamma = prob[i] < 0.5 ? confusionMatrix.getAdvNeg()
					: confusionMatrix.getAdvPos();
			nextLevelIndex = (int) Math.floor((number + gamma * prob[i]) * 2
					/ gamma);
			level.routeInstance(nextLevelIndex, instance, blnTrain);
		}
	}

	// private void addInstance(int index, Instance instance, boolean blnTrain)
	// {
	// if (children[index] == null)
	// children[index] = new AdaptiveMartiNode(new Instances(
	// inData.trainingInstances, 0), new Instances(
	// inData.holdoutInstances, 0), level + 1, index);
	// children[index].addInstance(instance, blnTrain);
	// }

	public void addInstance(Instance instance, boolean blnTrain) {
		if (blnTrain)
			inData.trainingInstances.add(instance);
		else
			inData.holdoutInstances.add(instance);
	}

	class NodeInputData {
		Instances trainingInstances;
		Instances holdoutInstances;
		double targetGammaMinus;
		double targetGammaPlus;

		public NodeInputData(Instances train, Instances holdout) {
			trainingInstances = new Instances(train);
			trainingInstances
					.setClassIndex(trainingInstances.numAttributes() - 1);
			holdoutInstances = new Instances(holdout);
			holdoutInstances
					.setClassIndex(holdoutInstances.numAttributes() - 1);
		}

		public int getTargetFN() {
			// TODO Auto-generated method stub
			return 0;
		}

		public int getTargetFP() {
			// TODO Auto-generated method stub
			return 0;
		}
	}

	class NodeOutputData {
		ConfusionMatrix trainConfusionMatrix;
		ConfusionMatrix holdoutConfusionMatrix;
		double[] trainProb;
		double[] holdoutProb;

		public NodeOutputData(int numTrainInstances, int numHoldoutInstances) {
			trainProb = new double[numTrainInstances];
			holdoutProb = new double[numHoldoutInstances];
		}
	}
}
