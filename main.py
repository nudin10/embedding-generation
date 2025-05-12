# DONT MOVE THIS FROM THE TOP. WILL BREAK ENV VARIABLE LOADING
from dotenv import load_dotenv
load_dotenv()

import logging
from tools.logger import Logger
from tools.cache import delete_huggingface_cache_directory
from slm.qwen import QwenSLM
from slm.phi import PhiSLM
from slm.slm import SLM
from tools.telegram_bot import send_error, send_message, send_warning
from tools.batch import batch_read_jsonl
from asyncio import run as async_run
from tools.storer import LocalEmbeddingStorer, S3EmbeddingStorer, LocalFileCleanupException
import json
import torch

async def main():
    logger = Logger(level=logging.DEBUG)

    qwen = QwenSLM
    phi = PhiSLM

    models = [
        {
            "model": qwen,
            "model_name": "Qwen"
        },
        {
            "model": phi,
            "model_name": "Phi"
        }
    ]

    for model_data in models:
        model_name = model_data["model_name"]

        await send_message(f"Started embedding generation: {model_name}")
        try:
            logger.info(f"Producing embeddings using {model_name}")

            try:
                model: SLM = model_data["model"](debug=False)
                # free unallocated memory
                torch.cuda.empty_cache()

            except Exception as e:
                message = f"Failed to load model {model_data['model_name']}: {e}"
                await send_error(message)
                logger.error(message)
                raise

            try:
                embedding_storer = S3EmbeddingStorer(False)

            except Exception as e:
                message = f"Failed to initialise S3 Embedding storer: {e}"
                await send_error(message)
                logger.error(message)
                raise

            try:
                RAW_DATA_PATH="./data/Magazine_Subscriptions.jsonl"
                batch_size = 70

                # # TODO: ONLY FOR TESTING. COMMENT WHEN DONE
                # limit = 10
                # batch_size = 5
                # limit_counter = 0

                for i, batch in enumerate(batch_read_jsonl(RAW_DATA_PATH, batch_size)):
                    await send_message(f"Producing {model_data['model_name']} embedding batch NO: {i+1}")

                    parsed_batch = []
                    metadata = []
                    for item in batch:
                        try:
                            parsed = json.loads(item)
                            
                            if parsed["text"] is not None:
                                parsed_batch.append(parsed["text"])
                            else:
                                parsed_batch.append("")

                            metadatum = {
                                "user_id": parsed["user_id"],
                                "item_id": parsed["asin"]
                            }
                            metadata.append(metadatum)

                        except json.JSONDecodeError as e:
                            message = f"Error decoding JSON: {e}"
                            await send_error(message)
                            logger.error(message)
                            continue

                        except Exception as e:
                            message = f"Unable to pre-produce texts from JSONL: {e}"
                            await send_error(message)
                            logger.error(message)
                            continue
                        

                    embeddings = model.generate_embeddings(parsed_batch)
                    df = embedding_storer.get_embeddings_dataframe(embeddings=embeddings, metadata=metadata)
                    embedding_storer.store_embeddings(df, f"./{model_name}_{i+1}", f"{model_name}_{i+1}")

                    # # TODO: ONLY FOR TESTING. COMMENT WHEN DONE
                    # limit_counter += 1
                    # if limit_counter == limit:
                    #     break
            
            except LocalFileCleanupException:
                message = f"Failed to cleanup local file: {model_name}_{i+1}.parquet"
                await send_warning(message)
                logger.error(message)
                # DO NOT RAISE. THIS IS MEANT AS A WARNING

            except Exception as e:
                message = f"Failed to generate embeddings: {model_name}"
                await send_error(message)
                logger.error(message)
                raise

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        finally:
            await send_message(f"Ended embedding generation: {model_name}")
            pass

        try:
            delete_huggingface_cache_directory()
        except Exception as e:
            message = f"Unable to clear huggingface cache: {e}"
            await send_message(message)
            logger.error(message)

if __name__ == "__main__":
    async_run(main())
