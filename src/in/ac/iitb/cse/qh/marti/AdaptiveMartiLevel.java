package in.ac.iitb.cse.qh.marti;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instance;

public class AdaptiveMartiLevel {
	private static final Logger LOGGER = Logger
			.getLogger(AdaptiveMartiLevel.class.getName());

	private int levelNumber;
	private List<AdaptiveMartiNode> nodes;
	private AdaptiveMartiLevel nextLevel;

	public AdaptiveMartiLevel(int level) {
		levelNumber = level;
		nodes = new ArrayList<AdaptiveMartiNode>();
	}

	public void addLevel() {
		nextLevel = new AdaptiveMartiLevel(levelNumber + 1);
	}

	public void addNode(AdaptiveMartiNode node) {
		if (nodes == null)
			nodes = new ArrayList<AdaptiveMartiNode>();
		node.setLevel(this);
		nodes.add(node);
	}

	public int getLevelNumber() {
		return levelNumber;
	}

	public void setLevelNumber(int levelNumber) {
		this.levelNumber = levelNumber;
	}

	public AdaptiveMartiLevel getNextLevel() {
		if (nextLevel == null)
			addLevel();
		return nextLevel;
	}

	public void setNextLevel(AdaptiveMartiLevel nextLevel) {
		this.nextLevel = nextLevel;
	}

	public List<AdaptiveMartiNode> getNodes() {
		return nodes;
	}

	public void build() throws Exception {
		for (AdaptiveMartiNode node : nodes)
			node.build();
	}

	public void routeInstance(int nextLevelIndex, Instance instance,
			boolean blnTrain) {
		AdaptiveMartiNode node = getNextLevel().getNode(nextLevelIndex);
		if (node == null) {
			node = AdaptiveMartiNodeFactory.createNode();
			node.setNumber(nextLevelIndex);
			getNextLevel().addNode(node);
		}
		node.addInstance(instance, blnTrain);
	}

	private AdaptiveMartiNode getNode(int nextLevelIndex) {
		AdaptiveMartiNode found = null;
		for (AdaptiveMartiNode node : getNodes())
			if (node.getNumber() == nextLevelIndex) {
				found = node;
				break;
			}
		return found;
	}

	public void evaluateLevel() {
		LOGGER.log(Level.INFO, "Number of nodes:" + nodes.size());
		for (AdaptiveMartiNode node : nodes)
			node.display();
	}
}
