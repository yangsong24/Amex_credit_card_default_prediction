import numpy as np
import pandas as pd

from typing import Dict

import pyspark
from pyspark import StorageLevel
from pyspark.sql import (
    SparkSession,
    types,
    functions as F,
)
from pyspark.sql.functions import (
    col,
    isnan,
    when,
    count,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoderModel,
    StringIndexerModel,
    VectorAssembler,
    ImputerModel,
)
from pyspark.ml.classification import (
    LogisticRegressionModel,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

import itertools

import pickle


## Create Spark Session
def get_spark_session():
    return SparkSession.builder.appName("amex-data").master("local[*]").getOrCreate()


def load_models(
    spark: pyspark.sql.session.SparkSession = None,
    indexer_path: str = "gs://icdp-bigdata-bucket/icdp_deployment/objects/indexer_model",
    imputer_path: str = "gs://icdp-bigdata-bucket/icdp_deployment/objects/imputer_model",
    ohe_path: str = "gs://icdp-bigdata-bucket/icdp_deployment/objects/ohe_model",
    va_path: str = "gs://icdp-bigdata-bucket/icdp_deployment/objects/va_model",
    lr_path: str = "gs://icdp-bigdata-bucket/icdp_deployment/objects/lr_model",
) -> Dict:

    if spark is None:
        raise ValueError("No SparkSession Provided!")

    return {
        "indexer_model": StringIndexerModel.load(indexer_path),
        "imputer_model": ImputerModel.load(imputer_path),
        "ohe_model": OneHotEncoderModel.load(ohe_path),
        "va_model": VectorAssembler.load(va_path),
        "lr_model": LogisticRegressionModel.load(lr_path),
    }


def load_meta_data(
    project_name: str = "big-data-86948",
    bucket_name: str = "icdp-bigdata-bucket",
    meta_data_path: str = "icdp_deployment/meta_data/meta_data.pkl",
) -> Dict:

    from google.cloud import storage

    storage_client = storage.Client(project=project_name)

    return pickle.loads(
        storage_client.bucket(bucket_name).blob(meta_data_path).download_as_string()
    )


def pipeline_pred(
    path: str = None,
    spark: SparkSession = None,
    models_dict: Dict = None,
    meta_data: Dict = None,
):
    if path == None:
        raise ValueError("Path Not Provided! Please Provide a Path")

    if spark is None:
        raise ValueError("No SparkSession Provided!")

    if models_dict is None:
        raise ValueError("models_dict Not Provided!")

    if meta_data is None:
        raise ValueError("Pipeline metadata not provided.")

    input_df = spark.read.option(
        "header",
        "True",
    ).csv(path, schema=meta_data["schema"]["train_schema"])

    input_df = input_df.drop(*meta_data["column_names"]["cols_to_drop"])
    input_df = models_dict["indexer_model"].transform(input_df)
    input_df = models_dict["imputer_model"].transform(input_df)
    input_df = models_dict["ohe_model"].transform(input_df)

    input_df = input_df.select(*meta_data["column_names"]["useful_cols"])

    new_num_cols = []
    for num_col in meta_data["column_names"]["num_cols_imputed"]:
        new_name = num_col.split("_")[0] + "_" + num_col.split("_")[1]
        new_num_cols.append(new_name)
        input_df = input_df.withColumnRenamed(num_col, new_name)
    new_cat_cols = []
    for cat_col in meta_data["column_names"]["cat_cols_ohe"]:
        new_name = cat_col.split("_")[0] + "_" + cat_col.split("_")[1]
        new_cat_cols.append(new_name)
        input_df = input_df.withColumnRenamed(cat_col, new_name)

    num_funcs = [
        (F.mean, "_mean"),
        (F.min, "_min"),
        (F.max, "_max"),
    ]

    cat_funcs = [
        (F.count, "_count"),
        (F.last, "_last"),
        (F.countDistinct, "_nunique"),
    ]

    agg_num_args = [
        func(col).alias(col + suffix)
        for col, (func, suffix) in itertools.product(new_num_cols, num_funcs)
    ]

    agg_cols_args = [
        func(col).alias(col + suffix)
        for col, (func, suffix) in itertools.product(new_cat_cols, cat_funcs)
    ]

    # Combine numeric and categoric agg arguments
    agg_args = agg_num_args + agg_cols_args

    input_df = input_df.groupBy("customer_ID").agg(*agg_args)

    input_df = (
        models_dict["va_model"].transform(input_df).select(["customer_ID", "features"])
    )

    out_df = models_dict["lr_model"].transform(input_df)

    output_cols = ["customer_ID", "rawPrediction", "probability", "prediction"]

    return out_df.select(*output_cols)
