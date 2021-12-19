// Databricks notebook source
// MAGIC %md
// MAGIC # Project NYSK clustering
// MAGIC 
// MAGIC ### Kmeans

// COMMAND ----------


import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd

import scala.xml._

import org.apache.hadoop.io.{ Text, LongWritable }
import org.apache.hadoop.conf.Configuration

import com.cloudera.datascience.common.XmlInputFormat
import com.cloudera.datascience.lsa.ParseWikipedia._
import com.cloudera.datascience.lsa.RunLSA._

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.{DenseMatrix => BDenseMatrix, DenseVector => BDenseVector, SparseVector => BSparseVector, Vector => BVector}
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.feature.StandardScaler



import java.io.StringReader
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;

import java.sql.Timestamp
import java.text.SimpleDateFormat

import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
import org.apache.spark.mllib.util.KMeansDataGenerator

import scala.collection.mutable.HashMap


// COMMAND ----------


def toBreeze(v:Vector) = BVector(v.toArray)
def fromBreeze(bv:BVector[Double]) = Vectors.dense(bv.toArray)
def add(v1:Vector, v2:Vector) = fromBreeze(toBreeze(v1) + toBreeze(v2))
def scalarMultiply(a:Double, v:Vector) = fromBreeze(a * toBreeze(v))

def stackVectors(v1:Vector, v2:Vector) = {
var v3 = Vectors.zeros(v1.size+v2.size)
    for (i <- 0 until v1.size) {
      BVector(v3.toArray)(i) = v1(i);
    }
    for (i <- 0 until v2.size) {
      BVector(v3.toArray)(v1.size+i) = v2(i);
    }
}

// SÃ©paration du fichier XML en un RDD oÃ¹ chaque Ã©lÃ©ment est un article
// Retourne un RDD de String Ã  partir du fichier "path"
def loadArticle(sc: SparkContext, path: String): RDD[String] = {
@transient val conf = new Configuration()
conf.set(XmlInputFormat.START_TAG_KEY, "<document>")
conf.set(XmlInputFormat.END_TAG_KEY, "</document>")
val in = sc.newAPIHadoopFile(path, classOf[XmlInputFormat], classOf[LongWritable], classOf[Text], conf)
in.map(line => line._2.toString)
}


// Pour un élément XML de type "document",
//   - on extrait le champ "date"
//   - on parse la chaÃ®ne de caractÃ¨re au format yyyy-MM-dd HH:mm:ss
//   - on retourne un Timestamp
def extractDate(elem: scala.xml.Elem): java.sql.Timestamp = {
    val dn: scala.xml.NodeSeq = elem \\ "date"
    val x: String = dn.text
    // d'aprÃ¨s l'exemple 2011-05-18 16:30:35
    val format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    if (x == "")
      return null
    else {
      val d = format.parse(x.toString());
      val t = new Timestamp(d.getTime());
      return t
    }
}

// Pour un élément XML de type "document",
//   - on extrait le champ #field
def extractString(elem: scala.xml.Elem, field: String): String = {
    val dn: scala.xml.NodeSeq = elem \\ field
    val x: String = dn.text
    return x
}

def extractInt(elem: scala.xml.Elem, field: String): Int = {
    val dn: scala.xml.NodeSeq = elem \\ field
    val x: Int = dn.text.toInt
    return x
}

def extractAll(elem: scala.xml.Elem, whatText: String = "text"): (Int, java.sql.Timestamp, String) = {
    return (extractInt(elem,"docid"), extractDate(elem), extractString(elem,whatText))
}

def extractText(elem: scala.xml.Elem): String = {
    return (extractString(elem,"title") + " " + extractString(elem,"summary") + " " + extractString(elem,"text"))
}

// NÃ©cessaire, car le type java.sql.Timestamp n'est pas ordonnÃ© par dÃ©faut (étonnant...)
implicit def ordered: Ordering[java.sql.Timestamp] = new Ordering[java.sql.Timestamp] {
def compare(x: java.sql.Timestamp, y: java.sql.Timestamp): Int = x compareTo y
}

def hasLetters(str: String): Boolean = {
// While loop for high performance
var i = 0
while (i < str.length) {
  if (Character.isLetter(str.charAt(i))) {
    return true
  }
  i += 1
}
false
}



// COMMAND ----------

val conf = new SparkConf().setAppName("NYSK").setMaster("local[*]")



conf.set("spark.hadoop.validateOutputSpecs", "false")
//conf.set("spark.driver.allowMultipleContexts", "true")
//conf.setMaster("local[*]")
//conf.set("spark.executor.memory", MAX_MEMORY)
//conf.set("spark.driver.memory", MAX_MEMORY)
//conf.set("spark.driver.maxResultSize", MAX_MEMORY)
//val sc = new SparkContext(conf)




// COMMAND ----------


    var textToExtract = "text";
    var useW2Vec = true;
    var weightTfIdf = true;
    println ("*****");

// COMMAND ----------

// MAGIC %md
// MAGIC ### Version avec XML

// COMMAND ----------


    val nysk_raw = loadArticle(sc, "dbfs:/FileStore/fichiers/nysk.xml")/*.sample(false,0.01)*/
    val nysk_xml: RDD[Elem] = nysk_raw.map(XML.loadString)

// COMMAND ----------

nysk_xml.take(10).foreach(println)

// COMMAND ----------


 val nyskxml: RDD[(Int, java.sql.Timestamp, String)] = nysk_xml.map(e => extractAll(e,textToExtract))
    val nyskTitlesxml: RDD[(Int, java.sql.Timestamp, String)] = nysk_xml.map(e => extractAll(e,"title"))
    val nyskSummariesxml: RDD[(Int, java.sql.Timestamp, String)] = nysk_xml.map(e => extractAll(e,"summary"))

// COMMAND ----------

//%fs mkdir -rf /FileStore/jars/

//dbutils.fs.rm("/databricks/driver/dbfs:/FileStore/shared_uploads/",true)
//display(dbutils.fs.ls("/FileStore/jars/"))



// COMMAND ----------

nyskxml.take(3).foreach(println)

// COMMAND ----------

    val stopwords = sc.textFile("dbfs:/FileStore/fichiers/stop_words").collect.toArray.toSet
    val stopwordsBroadcast = sc.broadcast(stopwords).value

// COMMAND ----------

 val lemmatizedWithDate = nyskxml.mapPartitions(iter => {
      val pipeline = com.cloudera.datascience.lsa.ParseWikipedia.createNLPPipeline();
      iter.map {
        case (docid, date, text) =>
          (docid.toString, date,
            com.cloudera.datascience.lsa.ParseWikipedia.plainTextToLemmas(text.toLowerCase.split("\\W+").mkString(" "), stopwordsBroadcast, pipeline))
      };
 })

// COMMAND ----------

lemmatizedWithDate.take(10).foreach(println)

// COMMAND ----------


    val lemmatized = lemmatizedWithDate.map { case (docid, date, text) => (docid, text) }
    val numTerms = 1000;

// COMMAND ----------

lemmatized.take(3).foreach(println)

// COMMAND ----------


val (termDocMatrix, termIds, docIds, idfs) = com.cloudera.datascience.lsa.ParseWikipedia.termDocumentMatrix(lemmatized, stopwordsBroadcast, numTerms, sc);
    



// COMMAND ----------

termDocMatrix.take(10).foreach(println)

// COMMAND ----------


val mat = new RowMatrix(termDocMatrix)
val k = 10 // nombre de valeurs singuliÃ¨res Ã  garder
val svd = mat.computeSVD(k, computeU=true)
val projections = mat.multiply(svd.V)
val projectionsTxt = projections.rows.map(l => l.toString.filter(c => c != '[' & c != ']'))
// Delete the existing path, ignore any exceptions thrown if the path doesn't exist
val outputProjection = "dbfs:/FileStore/shared_uploads/zarius3@free.fr/nysk/projection_LSA.txt"

//projectionsTxt.saveAsTextFile(outputProjection)

// COMMAND ----------

import org.apache.spark.mllib.clustering.KMeans
val nbClusters = 10
val nbIterations = 1000

val clustering = KMeans.train(termDocMatrix, nbClusters, nbIterations)
/*val outputClustering = "hdfs://head.local:9000/user/emeric/clusters"
try { hdfs.delete(new org.apache.hadoop.fs.Path(outputClustering), true) } 
catch { case _ : Throwable => { } }
clustering.save(sc, outputClustering)*/

val classes = clustering.predict(termDocMatrix)
val outputClasses = "dbfs:/FileStore/shared_uploads/zarius3@free.fr/nysk/classes_LSA.txt"
    
//classes.saveAsTextFile(outputClasses)

// COMMAND ----------


 
 val outputData = lemmatizedWithDate.zip(classes).map { case ((docid, date, title),cl) => (docid, date, cl) }.sortBy(_._2).map(l => l.toString.filter(c => c != '(' & c != ')'))
val outputDataFile = "dbfs:/FileStore/shared_uploads/zarius3@free.fr/nysk/output_LSA.txt"
     
outputData.saveAsTextFile(outputDataFile)

// COMMAND ----------


        clustering.clusterCenters.foreach(clusterCenter => {
            val highest = clusterCenter.toArray.zipWithIndex.sortBy(-_._1).map(v => v._2).take(10)
            println("*****")
            highest.foreach { s => print( termIds(s) + "," ) }
            println ()
            }
       )
