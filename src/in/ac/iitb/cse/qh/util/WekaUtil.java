package in.ac.iitb.cse.qh.util;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * A convenience class containing method to load weka instances from an ARFF
 * file
 * 
 * @author ashish
 * 
 */
public class WekaUtil {
	public static Instances getInstances(String file) throws Exception {
		DataSource datasource = new DataSource(file);
		Instances data = datasource.getDataSet();
		System.out.println("Class index is : " + data.classIndex());
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void balanceInstances(Instances instances) {
		int numInstancesPerClass[] = null;
		if (null != instances)
			numInstancesPerClass = instances.attributeStats(instances
					.classIndex()).nominalCounts;
		double r = (1.0d * numInstancesPerClass[1]) / numInstancesPerClass[0];

		for (Instance instance : instances)
			if (Utils.eq(instance.classValue(), 0.0))
				instance.setWeight(r);
	}

	public static int[][] classify(Logistic classifier, Instances instances,
			double[] prob) throws Exception {
		double[] dist = null;
		int instIndex = 0;
		int pred = 0;
		int[][] conf = new int[][] { { 0, 0 }, { 0, 0 } };
		for (Instance instance : instances) {
			dist = classifier.distributionForInstance(instance);
			prob[instIndex++] = dist[1];
			pred = dist[0] >= dist[1] ? 0 : 1;
			conf[(int) instance.classValue()][pred]++;
		}
		return conf;
	}
}
