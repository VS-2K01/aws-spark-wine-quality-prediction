from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, udf
from pyspark.sql.types import DoubleType

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Helper Functions

def prepare_labeled_df(df):
    # Cast numeric columns to double and rename 'quality' -> 'label' if present.
    df = df.select([col(c).cast("double").alias(c) for c in df.columns])
    if "quality" in df.columns and "label" not in df.columns:
        df = df.withColumnRenamed("quality", "label")
    return df


def majority_vote(pred1, pred2, pred3):
    # Majority vote over three predictions.
    # If any class gets at least 2 votes, return that class.
    # If all three predictions differ (no majority), return the rounded average of the three predictions.
    votes = [pred1, pred2, pred3]

    # Majority cases
    if votes.count(pred1) >= 2:
        return float(pred1)
    if votes.count(pred2) >= 2:
        return float(pred2)
    if votes.count(pred3) >= 2:
        return float(pred3)

    # No majority: use rounded average
    avg = (float(pred1) + float(pred2) + float(pred3)) / 3.0
    return float(round(avg))


majority_vote_udf = udf(majority_vote, DoubleType())


def main(test_path, model_base_path):
    spark = (
        SparkSession.builder
        .appName("WineQuality_Ensemble_Inference_v3")
        .getOrCreate()
    )

    print(f"---- Loading test data from: {test_path} ----")
    test_df_raw = spark.read.csv(test_path, header=True, inferSchema=True)
    test_df_raw = prepare_labeled_df(test_df_raw)

    base = model_base_path.rstrip("/")
    preprocess_model_path = base + "/preprocess"
    rf1_model_path = base + "/rf_model_1"
    rf2_model_path = base + "/rf_model_2"
    rf3_model_path = base + "/rf_model_3"

    print(f"=== Loading preprocessing model from: {preprocess_model_path} ===")
    preprocess_model = PipelineModel.load(preprocess_model_path)

    print("=== Transforming test data with preprocessing model ===")
    test_prepped = preprocess_model.transform(test_df_raw)

    # Add row_id to align predictions from each model
    test_prepped = test_prepped.withColumn("row_id", monotonically_increasing_id())

    print(f"=== Loading RF ensemble models from: {model_base_path} ===")
    rf1_model = RandomForestClassificationModel.load(rf1_model_path)
    rf2_model = RandomForestClassificationModel.load(rf2_model_path)
    rf3_model = RandomForestClassificationModel.load(rf3_model_path)

    print("=== Generating predictions from each RF model ===")
    pred_rf1 = rf1_model.transform(test_prepped) \
        .select("row_id", "label", col("prediction").alias("pred_rf1"))

    pred_rf2 = rf2_model.transform(test_prepped) \
        .select("row_id", col("prediction").alias("pred_rf2"))

    pred_rf3 = rf3_model.transform(test_prepped) \
        .select("row_id", col("prediction").alias("pred_rf3"))

    print("=== Computing ensemble prediction via majority vote / rounded average ===")
    ensemble_df = pred_rf1.join(pred_rf2, on="row_id") \
                          .join(pred_rf3, on="row_id")

    ensemble_df = ensemble_df.withColumn(
        "ensemble_prediction",
        majority_vote_udf(
            col("pred_rf1"),
            col("pred_rf2"),
            col("pred_rf3")
        )
    )

    print("=== Sample predictions from ensemble ===")
    cols_to_show = ["pred_rf1", "pred_rf2", "pred_rf3", "ensemble_prediction"]
    if "label" in ensemble_df.columns:
        cols_to_show = ["label"] + cols_to_show

    ensemble_df.select(cols_to_show).show(20, truncate=False)

    # Evaluate F1 if we have labels
    if "label" in ensemble_df.columns:
        print("=== Evaluating ensemble F1 on test data ===")
        eval_df = ensemble_df.select(
            "label",
            col("ensemble_prediction").alias("prediction")
        )

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        f1 = evaluator.evaluate(eval_df)
        print(f"Ensemble F1 score on test data: {f1}")
    else:
        print("No 'label' column found in test data. Skipping F1 evaluation.")

    print("=== Inference complete. Stopping Spark session. ===")
    spark.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python winePrediction_application.py <test_path> <model_base_path>")
        sys.exit(1)

    test_path = sys.argv[1]
    model_base_path = sys.argv[2]

    main(test_path, model_base_path)
