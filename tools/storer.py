import polars as pl
import logging
from tools.logger import Logger
import torch
import boto3
from datetime import datetime
import pytz
from pathlib import Path
import os


class Storer:
    def __init__(self):
        pass

    def flatten_embedding(self, embeddings):
        '''
        This is to flatten embeddings into a single list before storing into Polars Dataframe
        Generally more efficient than applying flattening after inserting tensor objects directly
        into Dataframes

        Tensors -> NumPy array 

        WARNING: THIS ESSENTIALLY COMPRESSES VECTORS INTO 1. THIS REMOVES ANY INDEX INFORMATION FOR EACH EMBEDDINGS
        '''

        try:
            np_array = embeddings.to(torch.float32).cpu().numpy()
            flattened_np_array = np_array.flatten().tolist()
            
            return flattened_np_array
        except Exception as e:
            self.logger.error(f"Error flattening embeddings: {e}")
            raise
    
    def get_embeddings_dataframe(self, embeddings: torch.Tensor, metadata:list) -> pl.DataFrame:
        '''
        Store embeddings in a Polars DataFrame
        Make sure metadata index matches the embeddings index
        '''
        if embeddings.shape[0] != len(metadata):
            self.logger.error(f"Shape mismatch: Embeddings batch size ({embeddings.shape[0]}) does not match metadata length ({len(metadata)})")
            raise ValueError("Batch size of embeddings must match length of metadata list.")
        
        batch_size = embeddings.shape[0]
        hidden_size = embeddings.shape[1]

        embeddings_np = embeddings.to(torch.float32).cpu().numpy()

        metadata_col_data = metadata

        embedding_col_data = [row.tolist() for row in embeddings_np]

        single_embedding_shape = (hidden_size,)
        ts_shape_col_data = [single_embedding_shape] * batch_size

        try:
            # column keys based on Amazon magazine subscription data
            data = pl.DataFrame({
                "metadata": metadata_col_data,         
                "embedding": embedding_col_data,        
                "ts_shape": ts_shape_col_data           
            })

            self.logger.debug(f"Created Polars DataFrame with shape: {data.shape}")
            return data

        except Exception as e:
            self.logger.error(f"Error creating Polars DataFrame from prepared data: {e}")
            raise


class LocalEmbeddingStorer(Storer):
    def __init__(self, debug=False):
        self.logger = Logger(name="EmbeddingStorer", level=logging.DEBUG if debug else logging.INFO)

    def store_embeddings(self, embeddings_df: pl.DataFrame, file_path):
        '''
        Store the DataFrame to a Parquet file
        '''
        
        try:
            embeddings_df.write_parquet(file_path)
            self.logger.debug(f"Embeddings stored successfully at {file_path}")
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {e}")
            raise

class LocalFileCleanupException(Exception):
    pass

class S3EmbeddingStorer(Storer):
    def __init__(self, debug=False):
        self.logger = Logger(name="S3EmbeddingStorer", level=logging.DEBUG if debug else logging.INFO)

        ACCESS_KEY=os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("RUNPOD_SECRET_AWS_ACCESS_KEY_ID")
        SECRET_KEY=os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY")
        if not ACCESS_KEY or not SECRET_KEY:
            self.logger.critical("AWS credentials not found in expected environment variables. Exiting...")
            exit(1)

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
        )

        self.bucket = "runpod-din"

        malaysia_timezone = pytz.timezone('Asia/Kuala_Lumpur')
        now = datetime.now(malaysia_timezone)
        
        self.s3_key_prefix = now.strftime('%d%m%Y') + "-"

    
    def store_embeddings(self, embeddings_df: pl.DataFrame, local_file_path, s3_object_name, cleanup=True):
        '''
        Store the DataFrame to an s3 bucket

        Provide file path without extension
        '''

        try:
            embeddings_df.write_parquet(local_file_path + ".parquet")
            self.logger.debug(f"Embeddings stored successfully at {local_file_path}")
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {e}")
            raise

        try:
            self.s3_client.upload_file(local_file_path + ".parquet", self.bucket, self.s3_key_prefix + s3_object_name + ".parquet")
            self.logger.debug(f"Embeddings uploaded successfully to s3")
        except Exception as e:
            self.logger.error(f"Error uploading embeedings to s3: {e}")
            raise
        
        if cleanup:
            file_to_delete = Path(local_file_path + ".parquet")
            try:
                file_to_delete.unlink(missing_ok=True)
            except Exception as e:
                raise LocalFileCleanupException
