# spark_batch_job.py
import os
import glob
import shutil
import logging
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.utils import AnalysisException

import pandas as pd

# ====== Import tiền xử lý ======
# File DataProcessing.py phải nằm trong PYTHONPATH / working dir

from DataProcessing import PreTrainingLayer

# ====== Config MinIO / S3A ======
MINIO_ENDPOINT = "http://minio:9000"  # Thêm protocol và port
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"

# Bucket/raw path
RAW_DATA_PATH = "s3a://aminer/"  # bucket aminer

# Local export
LOCAL_EXPORT_DIR = "/tmp/csv_export"  # yêu cầu của bạn
os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)


def list_s3a_files(spark: SparkSession, root_path: str) -> list[str]:
    """
    Duyệt liệt kê file trong S3A/MinIO bằng Hadoop FileSystem.
    Trả về danh sách uri đầy đủ (s3a://...).
    """
    try:
        # Cách an toàn để truy cập Java gateway
        jvm = spark._jvm
        jsc = spark._jsc

        # Cấu hình Hadoop
        hadoop_conf = jsc.hadoopConfiguration()
        hadoop_conf.set("fs.s3a.endpoint", MINIO_ENDPOINT)
        hadoop_conf.set("fs.s3a.access.key", MINIO_ACCESS_KEY)
        hadoop_conf.set("fs.s3a.secret.key", MINIO_SECRET_KEY)
        hadoop_conf.set("fs.s3a.path.style.access", "true")
        hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")

        # Tạo URI và Path
        uri = jvm.java.net.URI(root_path)
        path = jvm.org.apache.hadoop.fs.Path(root_path)
        fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)

        files = []

        # Kiểm tra nếu là file
        if fs.isFile(path):
            files.append(path.toString())
            return files

        # Nếu là thư mục, duyệt đệ quy
        from py4j.java_gateway import java_import
        java_import(jvm, "org.apache.hadoop.fs.*")

        remote_iterator = fs.listFiles(path, True)  # True = recursive
        while remote_iterator.hasNext():
            file_status = remote_iterator.next()
            if file_status.isFile():
                file_path = file_status.getPath().toString()
                # Chỉ lấy file JSON
                if file_path.lower().endswith((".json", ".ndjson")):
                    files.append(file_path)

        return files

    except Exception as e:
        logging.error(f"Error listing S3A files: {e}")
        # Fallback: sử dụng glob pattern nếu list files thất bại
        try:
            df = spark.read.json(f"{root_path}*.json")
            file_paths = [f.path for f in df.inputFiles()]
            return file_paths
        except Exception as fallback_error:
            logging.error(f"Fallback method also failed: {fallback_error}")
            return []  # Luôn trả về list rỗng thay vì None




def apply_pretrainlayer_map_in_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Nhận 1 partition dạng pandas.DataFrame, áp dụng tiền xử lý
    """
    processor = PreTrainingLayer()

    # Xử lý từng hàng
    texts = []
    ts_first = []
    labels = []

    for _, row in pdf.iterrows():
        try:
            # Chuyển row thành dict
            row_dict = row.to_dict()
            tup = processor.format_sample_to_text(row_dict)

            # Xử lý text
            text = tup[0] if tup and len(tup) > 0 else ""
            texts.append(text)

            # Xử lý timestamp
            ts_val = tup[1] if tup and len(tup) > 1 else None
            ts0 = None
            if isinstance(ts_val, (list, tuple)) and len(ts_val) > 0:
                ts0 = float(ts_val[0]) if ts_val[0] is not None else None
            elif isinstance(ts_val, (int, float)):
                ts0 = float(ts_val)
            ts_first.append(ts0)

            # Xử lý label
            label = row_dict.get('label') if 'label' in row_dict else None
            labels.append(label)

        except Exception as e:
            logging.error(f"Error processing row: {e}")
            texts.append("")
            ts_first.append(None)
            labels.append(None)

    out = pd.DataFrame({
        "text_representation": texts,
        "timestamp": ts_first,
        "label": labels
    })

    return out


def process_one_file(spark: SparkSession, input_uri: str):
    """
    Đọc một file từ S3A, áp dụng tiền xử lý, và export CSV
    """
    logging.info(f"Processing file: {input_uri}")

    try:
        # Heuristic đọc file
        lower = input_uri.lower()
        if lower.endswith((".json", ".ndjson")):
            # Đọc với chế độ cho phép lỗi
            df_raw = (spark.read
                      .option("multiLine", "false")
                      .option("mode", "PERMISSIVE")  # Cho phép lỗi
                      .option("columnNameOfCorruptRecord", "_corrupt_record")  # Đặt tên cột lỗi
                      .json(input_uri))

        else:
            logging.warning(f"Skip unsupported file type: {input_uri}")
            return

        # CACHE DataFrame trước khi thực hiện bất kỳ truy vấn nào trên _corrupt_record
        df_raw = df_raw.cache()

        # Kiểm tra xem có cột _corrupt_record không
        has_corrupt_column = "_corrupt_record" in df_raw.columns

        if has_corrupt_column:
            # Đếm số bản ghi lỗi (sau khi cache)
            corrupt_count = df_raw.filter(df_raw["_corrupt_record"].isNotNull()).count()
            if corrupt_count > 0:
                logging.warning(f"Found {corrupt_count} corrupt records in {input_uri}")
                # Lọc bỏ các bản ghi lỗi
                df_raw = df_raw.filter(df_raw["_corrupt_record"].isNull())

            # Bỏ cột _corrupt_record
            df_raw = df_raw.drop("_corrupt_record")

        # Nếu không có dòng nào thì bỏ qua
        if df_raw.rdd.isEmpty():
            logging.warning(f"No valid data in {input_uri}, skipping.")
            df_raw.unpersist()  # Giải phóng cache
            return

        # Schema đầu ra cho mapInPandas
        out_schema = StructType([
            StructField("text_representation", StringType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("label", StringType(), True),
        ])

        # Áp dụng PreTrainingLayer
        df_processed = df_raw.mapInPandas(apply_pretrainlayer_map_in_pandas, schema=out_schema)

        # Add source file column để trace
        df_processed = df_processed.withColumn("source_file", lit(input_uri))

        # Ghi ra local
        base = os.path.basename(input_uri)
        base_no_ext = os.path.splitext(base)[0]
        temp_dir = os.path.join(LOCAL_EXPORT_DIR, f"tmp_{base_no_ext}")
        final_csv = os.path.join(LOCAL_EXPORT_DIR, f"{base_no_ext}.csv")

        # Dọn temp nếu còn
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Ghi 1 file (coalesce(1)) + header
        (df_processed.coalesce(1)
         .write.mode("overwrite")
         .option("header", "true")
         .csv(temp_dir))

        # Move part-*.csv → <basename>.csv
        part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
        if not part_files:
            logging.warning(f"No CSV writer output for {input_uri}")
        else:
            # Nếu final_csv tồn tại → ghi đè
            if os.path.exists(final_csv):
                os.remove(final_csv)
            shutil.move(part_files[0], final_csv)
            logging.info(f"Saved: {final_csv}")

        # cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Giải phóng cache
        df_raw.unpersist()

    except Exception as e:
        logging.error(f"Error processing file {input_uri}: {e}")
        # Đảm bảo giải phóng cache ngay cả khi có lỗi
        try:
            if 'df_raw' in locals():
                df_raw.unpersist()
        except:
            pass


def main():
    try:
        spark = (
            SparkSession.builder
            .appName("PretrainLayerBatchExport")
            # .config("spark.jars",
            #         "/opt/bitnami/spark/jars/hadoop-aws-3.3.4.jar,/opt/bitnami/spark/jars/aws-java-sdk-bundle-1.12.262.jar")
            # .config("spark.driver.extraClassPath",
            #         "/opt/bitnami/spark/jars/hadoop-aws-3.3.4.jar:/opt/bitnami/spark/jars/aws-java-sdk-bundle-1.12.262.jar")
            # .config("spark.executor.extraClassPath",
            #         "/opt/bitnami/spark/jars/hadoop-aws-3.3.4.jar:/opt/bitnami/spark/jars/aws-java-sdk-bundle-1.12.262.jar")
            .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
            .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            # .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            #         "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .getOrCreate()
        )

        # Liệt kê file trong RAW_DATA_PATH
        logging.info(f"Listing files in {RAW_DATA_PATH}")
        input_files = list_s3a_files(spark, RAW_DATA_PATH)

        # Đảm bảo input_files không phải None
        if input_files is None:
            input_files = []
            logging.warning("list_s3a_files returned None, using empty list")

        if not input_files:
            logging.warning(f"No JSON files found under {RAW_DATA_PATH}")
            # Thử phương pháp thay thế: đọc trực tiếp với pattern
            try:
                df_test = spark.read.json(f"{RAW_DATA_PATH}*.json")
                input_files = [f.path for f in df_test.inputFiles()]
                logging.info(f"Found {len(input_files)} files using pattern matching")
            except Exception as e:
                logging.error(f"Alternative method also failed: {e}")
                input_files = []  # Đảm bảo là list rỗng
        else:
            logging.info(f"Found {len(input_files)} JSON files under {RAW_DATA_PATH}")

        # Xử lý từng file
        for i, f in enumerate(input_files, 1):
            logging.info(f"Processing file {i}/{len(input_files)}: {f}")
            process_one_file(spark, f)

        spark.stop()
        logging.info("Done. All files processed.")

    except AnalysisException as e:
        logging.error(f"Spark operation failed: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Batch job failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()