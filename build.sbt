name := "QuoraQuestionPairs"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.10" % "2.2.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "2.2.0"
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0"
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0" classifier "models"
libraryDependencies += "com.typesafe" % "config" % "1.3.1"