import asyncio
import os
import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.grpc.aio as aio_triton_grpc

from typing import List


class EmbeddingExtractor:
    """
    Uses NVIDIA Triton Inference Server to extract embeddings from text.
    """
    client = triton_grpc.InferenceServerClient(
        os.getenv("TRITON_HOST", "192.168.7.50:8001")
    )
    aio_client = aio_triton_grpc.InferenceServerClient(
        os.getenv("TRITON_HOST", "192.168.7.50:8001")
    )

    @classmethod
    def get_embeddings(
        cls, texts: List[str], model_name: str = "embedding", batch_size: int = 8
    ) -> np.ndarray:
        embeddings = []
        for batch_i in range(0, len(texts), batch_size):
            batch_texts = texts[batch_i : batch_i + batch_size]
            batch_embeddings = cls.__request_batch(batch_texts, model_name)
            embeddings.append(batch_embeddings)
        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    @classmethod
    def __request_batch(
        cls, batch_texts: List[str], model_name: str = "embedding"
    ) -> np.ndarray:
        input_name = "INPUT"
        inputs = [triton_grpc.InferInput(input_name, [len(batch_texts), 1], "BYTES")]

        input_array = np.array([str(text).encode("UTF-8") for text in batch_texts])
        input_array = np.expand_dims(input_array, axis=1)
        inputs[0].set_data_from_numpy(input_array)

        output_name = "OUTPUT"
        outputs = [triton_grpc.InferRequestedOutput(output_name)]

        response = cls.client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        
        if response is None:
            raise ValueError("Inference response is None")

        embeddings = response.as_numpy(output_name)
        if embeddings is None:
            raise ValueError("Failed to retrieve embeddings from the response")

        return embeddings

    @classmethod
    async def async_get_embeddings(
        cls, texts: List[str], model_name: str = "embedding", batch_size: int = 8
    ) -> np.ndarray:
        tasks = []
        for batch_i in range(0, len(texts), batch_size):
            batch_texts = texts[batch_i : batch_i + batch_size]
            tasks.append(
                asyncio.create_task(cls.__async_request_batch(batch_texts, model_name))
            )
        embeddings = np.concatenate(await asyncio.gather(*tasks), axis=0)

        return embeddings

    @classmethod
    async def __async_request_batch(
        cls, batch_texts: List[str], model_name: str = "embedding"
    ) -> np.ndarray:
        input_name = "INPUT"
        inputs = [
            aio_triton_grpc.InferInput(input_name, [len(batch_texts), 1], "BYTES")
        ]

        input_array = np.array([str(text).encode("UTF-8") for text in batch_texts])
        input_array = np.expand_dims(input_array, axis=1)
        inputs[0].set_data_from_numpy(input_array)

        output_name = "OUTPUT"
        outputs = [aio_triton_grpc.InferRequestedOutput(output_name)]

        response = await cls.aio_client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        if response is None:
            raise ValueError("Inference response is None")

        embeddings = response.as_numpy(output_name)
        if embeddings is None:
            raise ValueError("Failed to retrieve embeddings from the response")

        return embeddings
    
