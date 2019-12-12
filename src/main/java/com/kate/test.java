package com.kate;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import static org.apache.spark.sql.functions.*;

public class test implements Serializable{

	private final static String FILE_DATA = "/Users/brestbiblechurch/IdeaProjects/decisiontree/src/main/java/com/kate/DEFAULT_PUBLIC_CUSTOMER.csv";

	public static void main(String[] args) {
				test testDecisionTree = new test();
		testDecisionTree.run();
	}

	public void run() {

		SparkSession spark = SparkSession.builder().master("local").appName("TestDecisionTreeGermanBank")
				.config("spark.sql.warehouse.dir", "working").getOrCreate();
		
		// Loads data.
		Dataset<Row> datasetCredit = spark.read().schema(buildSchema()).csv(FILE_DATA);
		datasetCredit = datasetCredit.select(
			col("name").as("name"),
			col("age").as("age"),
			col("include").as("include")
				);
		
		// Show some data
		datasetCredit.show();
		
		// Verify Schema
		datasetCredit.printSchema();
		
		datasetCredit.describe("name").show();
		datasetCredit.describe("age").show();
		datasetCredit.describe("include").show();
		//datasetCredit.groupBy(col("creditability")).agg(avg("balance"),  avg("amount"), avg("duration"), count("*")).show();
		
		// Decision Tree algo
	    // Split data 70% and 30%
	    Dataset<Row>[] datas = datasetCredit.randomSplit(new double[] {0.7, 0.3});
	    
	    // separate dataset
	    JavaRDD<Row> train = datas[0].toJavaRDD();
	    JavaRDD<Row> test = datas[1].toJavaRDD();

	    
	    // Training data
	    @SuppressWarnings("serial")
		JavaRDD<LabeledPoint> trainLabeledPoints = train.map(
				(Function<Row, LabeledPoint>) row -> new LabeledPoint(row.getDouble(2), // creditability
					Vectors.dense(
							row.getInt(0),
							row.getInt(1)
							)));
	    
	    trainLabeledPoints.cache();
	    
	    // Validation data
	    @SuppressWarnings("serial")
		JavaRDD<LabeledPoint> testLabeledPoints = test.map(
				(Function<Row, LabeledPoint>) row -> new LabeledPoint(row.getDouble(2), // creditability
					Vectors.dense(
							row.getInt(0),
							row.getInt(1)
							)));
	    testLabeledPoints.cache();
		
	    // TODO a tester, les variables non-continues doivent etre liste dans ce param
	    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
	    String impurity = "gini";
	    Integer maxDepth = 3;
	    Integer maxBins = 20;
	    int numClass = 2;
	    
	    // Find a model ...
	    final DecisionTreeModel model = DecisionTree.trainClassifier(trainLabeledPoints, numClass, categoricalFeaturesInfo, impurity, maxDepth, maxBins);
	    
	    System.out.println("--------------- MODEL ---------------------" );
	    System.out.println(model.toDebugString());
	    System.out.println("------------------------------------" );
	    
	    // Evaluate model on test instances and compute test error
	    JavaPairRDD<Double, Double> predictionAndLabel =
	    		testLabeledPoints.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
	    double testErr =
	      predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testLabeledPoints.count();

	    System.out.println("Test Error: " + testErr);
	    System.out.println("Learned classification tree model:\n" + model.toDebugString());
	}
	
	protected StructType buildSchema() {
	    return new StructType(
	        new StructField[] {
		        DataTypes.createStructField("name", DataTypes.IntegerType, true),
		        DataTypes.createStructField("age", DataTypes.IntegerType, true),
		        DataTypes.createStructField("include", DataTypes.DoubleType, true) });
	}


}
