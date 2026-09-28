from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



# Helper functions

def prepare_labeled_df(df):
    # Cast all columns to double and rename 'quality' to 'label'.
    df = df.select([col(c).cast("double").alias(c) for c in df.columns])
    if "quality" in df.columns and "label" not in df.columns:
        df = df.withColumnRenamed("quality", "label")
    return df


def build_preprocess_pipeline(feature_cols):
    # Build a pipeline that: Assembles features into 'features' and scales them into 'scaledFeatures'
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )
    return Pipeline(stages=[assembler, scaler])


def add_class_weights(df, label_col="label", weight_col="classWeight"):
    # Compute class weights from df: weight(label) = total_count / count(label) and add a weight column.
    label_counts = df.groupBy(label_col).count().collect()
    total_count = df.count()

    weight_dict = {
        row[label_col]: float(total_count) / row["count"]
        for row in label_counts
    }

    spark = df.sql_ctx.sparkSession
    broadcast_weights = spark.sparkContext.broadcast(weight_dict)

    def get_weight(label):
        return float(broadcast_weights.value.get(label, 1.0))

    weight_udf = udf(get_weight, DoubleType())
    df_weighted = df.withColumn(weight_col, weight_udf(col(label_col)))
    return df_weighted


# Main training logic

def main(train_path, val_path, model_output_path):
    spark = (
        SparkSession.builder
        .appName("WineQuality_RF_ThreeModels_FixedHyperparams")
        .getOrCreate()
    )

    # Load data
    train_df_raw = spark.read.csv(train_path, header=True, inferSchema=True)
    val_df_raw = spark.read.csv(val_path, header=True, inferSchema=True)

    # Prepare (cast to double, rename quality -> label)
    train_df_raw = prepare_labeled_df(train_df_raw)
    val_df_raw = prepare_labeled_df(val_df_raw)

    # Build and fit preprocessing on TRAIN only
    feature_cols = [c for c in train_df_raw.columns if c != "label"]
    preprocess_pipeline = build_preprocess_pipeline(feature_cols)
    preprocess_model = preprocess_pipeline.fit(train_df_raw)

    # Transform train and val with same preprocessing
    train_prepped = preprocess_model.transform(train_df_raw)
    val_prepped = preprocess_model.transform(val_df_raw)

    # Add class weights on the full (imbalanced) training data
    train_weighted = add_class_weights(train_prepped, label_col="label", weight_col="classWeight")

    # Define the three RF classifiers with your hyperparameters
    rf1 = RandomForestClassifier(
        labelCol="label",
        featuresCol="scaledFeatures",
        weightCol="classWeight",
        numTrees=200,
        maxDepth=10,
        seed=42
    )

    rf2 = RandomForestClassifier(
        labelCol="label",
        featuresCol="scaledFeatures",
        weightCol="classWeight",
        numTrees=200,
        maxDepth=10,
        maxBins=64,
        minInstancesPerNode=1,
        featureSubsetStrategy="auto",
        seed=423
    )

    rf3 = RandomForestClassifier(
        labelCol="label",
        featuresCol="scaledFeatures",
        weightCol="classWeight",
        numTrees=200,
        maxDepth=10,
        maxBins=96,
        minInstancesPerNode=1,
        featureSubsetStrategy="auto",
        seed=17
    )

    print("Training RF model 1...")
    rf1_model = rf1.fit(train_weighted)

    print("Training RF model 2...")
    rf2_model = rf2.fit(train_weighted)

    print("Training RF model 3...")
    rf3_model = rf3.fit(train_weighted)

    # Evaluate each model on the validation set
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    preds_rf1 = rf1_model.transform(val_prepped)
    f1_rf1 = evaluator.evaluate(preds_rf1)

    preds_rf2 = rf2_model.transform(val_prepped)
    f1_rf2 = evaluator.evaluate(preds_rf2)

    preds_rf3 = rf3_model.transform(val_prepped)
    f1_rf3 = evaluator.evaluate(preds_rf3)

    print("---- Validation F1 scores for the three RF models ----")
    print(f"RF Model 1 (maxBins=32, seed=42)  F1: {f1_rf1:.4f}")
    print(f"RF Model 2 (maxBins=64, seed=423) F1: {f1_rf2:.4f}")
    print(f"RF Model 3 (maxBins=96, seed=17)  F1: {f1_rf3:.4f}")

    # 8. Save preprocessing + all three RF models
    base = model_output_path.rstrip("/")
    preprocess_model_path = base + "/preprocess"
    rf1_model_path = base + "/rf_model_1"
    rf2_model_path = base + "/rf_model_2"
    rf3_model_path = base + "/rf_model_3"

    print(f"Saving preprocessing model to: {preprocess_model_path}")
    preprocess_model.write().overwrite().save(preprocess_model_path)

    print(f"Saving RF model 1 to: {rf1_model_path}")
    rf1_model.write().overwrite().save(rf1_model_path)

    print(f"Saving RF model 2 to: {rf2_model_path}")
    rf2_model.write().overwrite().save(rf2_model_path)

    print(f"Saving RF model 3 to: {rf3_model_path}")
    rf3_model.write().overwrite().save(rf3_model_path)

    print("Training run complete. Stopping Spark.")
    spark.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print(
            "Usage: spark-submit winePrediction_model.py "
            "<train_path> <val_path> <model_output_base_path>"
        )
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    model_output_path = sys.argv[3]

    main(train_path, val_path, model_output_path)
