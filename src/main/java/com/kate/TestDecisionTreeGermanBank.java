package com.kate;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.linalg.Vectors;

import static org.apache.spark.sql.functions.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class TestDecisionTreeGermanBank implements Serializable{

	private final static String FILE_DATA = "/Users/brestbiblechurch/IdeaProjects/firstXgboost/src/main/java/com/kate/data.csv";


	public static void main(String[] args) {
				TestDecisionTreeGermanBank testDecisionTree = new TestDecisionTreeGermanBank();
		testDecisionTree.run();
	}

	public void run() {

		SparkSession spark = SparkSession.builder().master("local").appName("TestDecisionTreeGermanBank")
				.config("spark.sql.warehouse.dir", "working").getOrCreate();
		
		// Loads data.
		Dataset<Row> datasetCredit = spark.read().schema(buildSchema()).csv(FILE_DATA);
		datasetCredit = datasetCredit.select(
			col("creditability"),
			col("balance#").minus(lit(1)).as("balance"),
			col("duration"),
			col("history"),
			col("purpose"),
			col("amount"),
			col("savings#").minus(lit(1)).as("savings"),
			col("employment#").minus(lit(1)).as("employment"),
			col("instPercent"), //8
			col("sexMarried#").minus(lit(1)).as("sexMarried"),
			col("guarantors#").minus(lit(1)).as("guarantors"), //10
			col("residenceDuration#").minus(lit(1)).as("residenceDuration"),
			col("assets#").minus(lit(1)).as("assets"), //12
			col("age"), 
			col("concCredit#").minus(lit(1)).as("concCredit"),
			col("apartment#").minus(lit(1)).as("apartment"), 
			col("credits#").minus(lit(1)).as("credits"),
			col("occupation#").minus(lit(1)).as("occupation"),
			col("dependents#").minus(lit(1)).as("dependents"), 
			col("hasPhone#").minus(lit(1)).as("hasPhone"),
			col("foreign#").minus(lit(1)).as("foreign")
				);
		
		// Show some data
		datasetCredit.show();
		
		// Verify Schema
		datasetCredit.printSchema();
		
		datasetCredit.describe("amount").show();
		datasetCredit.describe("balance").show();
		datasetCredit.groupBy(col("creditability")).agg(avg("balance"),  avg("amount"), avg("duration"), count("*")).show();
		
		// Decision Tree algo
	    // Split data 70% and 30%
	    Dataset<Row>[] datas = datasetCredit.randomSplit(new double[] {0.7, 0.3});
	    
	    // separate dataset
	    JavaRDD<Row> train = datas[0].toJavaRDD();
	    JavaRDD<Row> test = datas[1].toJavaRDD();

	    
	    // Training data
	    @SuppressWarnings("serial")
		JavaRDD<LabeledPoint> trainLabeledPoints = train.map(
	    		new Function<Row,LabeledPoint>() {
	    		             @Override
	    		            public LabeledPoint call(Row row) throws Exception {
	    		            	
	    		                return new LabeledPoint(row.getDouble(0), // creditability
	    		                	Vectors.dense(
	    		                			row.getDouble(1), 
	    		                			row.getDouble(2), 
	    		                			row.getDouble(3), 
	    		                			row.getDouble(4), 
	    		                			row.getDouble(5), 
	    		                			row.getDouble(6),
	    		                			row.getDouble(7), 
	    		                			row.getDouble(8), 
	    		                			row.getDouble(9), 
	    		                			row.getDouble(10), 
	    		                			row.getDouble(11),
	    		                			row.getDouble(12), 
	    		                			row.getDouble(13), 
	    		                			row.getDouble(14), 
	    		                			row.getDouble(15), 
	    		                			row.getDouble(16),
	    		                			row.getDouble(17), 
	    		                			row.getDouble(18), 
	    		                			row.getDouble(19), 
	    		                			row.getDouble(20) 
	    		                			));
	    		            }
	    		        });
	    
	    trainLabeledPoints.cache();
	    
	    // Validation data
	    @SuppressWarnings("serial")
		JavaRDD<LabeledPoint> testLabeledPoints = test.map(
	    		new Function<Row,LabeledPoint>() {
	    		             @Override
	    		            public LabeledPoint call(Row row) throws Exception {
	    		            	
	    		                return new LabeledPoint(row.getDouble(0), // creditability
	    		                	Vectors.dense(
	    		                			row.getDouble(1), 
	    		                			row.getDouble(2), 
	    		                			row.getDouble(3), 
	    		                			row.getDouble(4), 
	    		                			row.getDouble(5), 
	    		                			row.getDouble(6),
	    		                			row.getDouble(7), 
	    		                			row.getDouble(8), 
	    		                			row.getDouble(9), 
	    		                			row.getDouble(10), 
	    		                			row.getDouble(11),
	    		                			row.getDouble(12), 
	    		                			row.getDouble(13), 
	    		                			row.getDouble(14), 
	    		                			row.getDouble(15), 
	    		                			row.getDouble(16),
	    		                			row.getDouble(17), 
	    		                			row.getDouble(18), 
	    		                			row.getDouble(19), 
	    		                			row.getDouble(20) 
	    		                			));
	    		            }
	    		        });
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
	            DataTypes.createStructField("creditability", DataTypes.DoubleType, true),
	            DataTypes.createStructField("balance#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("duration", DataTypes.DoubleType, true),
		        DataTypes.createStructField("history", DataTypes.DoubleType, true),
		        DataTypes.createStructField("purpose", DataTypes.DoubleType, true),
		        DataTypes.createStructField("amount", DataTypes.DoubleType, true),
		        DataTypes.createStructField("savings#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("employment#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("instPercent", DataTypes.DoubleType, true),
		        DataTypes.createStructField("sexMarried#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("guarantors#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("residenceDuration#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("assets#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("age", DataTypes.DoubleType, true),
		        DataTypes.createStructField("concCredit#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("apartment#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("credits#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("occupation#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("dependents#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("hasPhone#", DataTypes.DoubleType, true),
		        DataTypes.createStructField("foreign#", DataTypes.DoubleType, true) });
	}


}
