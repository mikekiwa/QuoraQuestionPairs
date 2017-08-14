package QuestionPairClassifier

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline

import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.RandomForestClassifier
import com.typesafe.config._
import org.apache.log4j.{Level, LogManager, Logger}

object classifier {

  def processData(df: DataFrame): DataFrame = {
    val toInt = udf((s: String) => {
      s match {
        case "0" => 0
        case "1" => 1
        case _ => 0
      }
    })

    val data = df.na.drop()
      .withColumn("label", toInt(df("is_duplicate")))

    //data.select("id", "question1", "question2", "is_duplicate", "label").show(5)

    data
  }

  def filterData(trainData: DataFrame): DataFrame = {
    val t1 = new Tokenizer().setInputCol("question1").setOutputCol("q1_words")
    val t2 = new Tokenizer().setInputCol("question2").setOutputCol("q2_words")

    val stopwords: Array[String] = StopWordsRemover.loadDefaultStopWords("english")

    val filtered1 = new StopWordsRemover().setStopWords(stopwords).setCaseSensitive(false).
      setInputCol("q1_words").setOutputCol("q1_filtered")
    val filtered2 = new StopWordsRemover().setStopWords(stopwords).setCaseSensitive(false).
      setInputCol("q2_words").setOutputCol("q2_filtered")

    val pipeline = new Pipeline().setStages(Array(t1, t2, filtered1, filtered2))

    val model = pipeline.fit(trainData)
    val df = model.transform(trainData)

    // Calculate % of words that match between question1 and question2
    val getRatio = udf((q1: WrappedArray[String], q2: WrappedArray[String]) => {
      if (q1.isEmpty && q2.isEmpty) Vectors.dense(0.0, 0.0)
      else {
        val shared_q1 = q1.toArray.filter(word => q2.contains(word))
        val shared_q2 = q2.toArray.filter(word => q1.contains(word))
        val ratio = (shared_q1.length.toDouble + shared_q2.length.toDouble) / (q1.length.toDouble + q2.length.toDouble)
        Vectors.dense(ratio, shared_q1.length.toDouble)
      }
    })

    val trainDF = df
      .withColumn("word_match_ratio", getRatio(df("q1_filtered"), df("q2_filtered")))
      .drop("q1_words")
      .drop("q2_words")

    trainDF
  }

  def getModel(df: DataFrame) = {
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("word_match_ratio")
      .setNumTrees(10)

    val rfModel = rf.fit(df)
    rfModel
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val props = ConfigFactory.load()
    val envProps = props.getConfig(args(0))
    val spark = SparkSession
      .builder()
      .appName("Quora Question Pairs Similarity")
      .master(envProps.getString("executionMode"))
      .getOrCreate()
    val sc = spark.sparkContext

    spark.sparkContext.setLogLevel("ERROR")

    LogManager.getRootLogger.setLevel(Level.ERROR)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val traindf = spark.read.option("header", "true").csv("data/train.csv")
    val testdf = spark.read.option("header", "true").csv("data/test.csv")

    val trainData = processData(traindf)
    val testData = testdf

    // Train RandomForest model
    val train_df = filterData(trainData)
    val rfModel = getModel(train_df)

    // Make predictions on test data
    val testDF = filterData(testData)
    val predictions = rfModel.transform(testDF)

    val toStr = udf((s: String) => {
      s match {
        case "0.0" => "0"
        case "1.0" => "1"
        case _ => "0"
      }
    })

    val pred_data = predictions.withColumn("is_duplicate", toStr(predictions.select("prediction")("prediction")))

    val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

    predictions.withColumn("is_duplicate", stringify(predictions.select("prediction")("prediction"))).
      write.format("csv").save("/Users/ashwinvprabhu/Documents/workspace/Scala/QuoraQuestionPairs/data/out")
    //val pred_data = predictions.withColumn("is_duplicate", toStr(predictions.select("prediction")("prediction")))

    //pred_data.select("is_duplicate").write.format("csv").save("/Users/ashwinvprabhu/Documents/workspace/Scala/QuoraQuestionPairs/data/out ")
  }

}
