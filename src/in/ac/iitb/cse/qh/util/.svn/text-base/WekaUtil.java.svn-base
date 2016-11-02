package in.ac.iitb.cse.qh.util;

import weka.core.Instances;
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
}
