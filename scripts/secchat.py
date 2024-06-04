"""
Simple command line application to chat with a secondary project using an LLM.
"""

from typing import Dict, List, Tuple, Callable, Optional
import os

import accelerate  # appears to be unused but speeds up model loading
import llama_cpp
import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer

from projects import emb
from projects import config
from projects import secdataset as sd
from projects import secondary
from projects import util

MAX_RECORDS = 5
MAX_CONTEXT_TOKENS = 4096
LLAMA_CPP_VERBOSE = False

DEVICE_MPS = 'mps'

E5_SMALL_NAME = 'intfloat/e5-small-v2'
MPNET_NAME = 'sentence-transformers/all-mpnet-base-v2'

LLAMA3_8B_INSTRUCT = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'
MISTRAL_7B_INSTRUCT_03 = 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'


def main():
    """Main program!"""

    # ~~~~ configuration ~~~~

    seed = 0
    use_e5 = False
    window_size = 4
    window_step = window_size // 2

    # currently hard-coded with my personal worklog
    input_dir_path = '/Users/Shared/writing/worklog/content'
    input_file_names = [
        'master.sec',
        'gamedev.sec',
        'info.sec'
    ]

    # ~~~~ calculate / load embeddings ~~~~

    items = {}

    for input_file_name in input_file_names:
        input_file_path = os.path.join(input_dir_path, input_file_name)
        content = ''.join(util.read_lines(input_file_path))

        items_cur = secondary.parse_secondary_file(content)
        # toss out reference items
        items_cur = [x for x in items_cur if 'id' in x]
        items_cur_dict = {x['id']: x for x in items_cur}

        items = {**items, **items_cur_dict}

    records_paragraphs, records_windows, missing_notes = sd.sec_records_paragraphs(
        items=items,
        window_size=window_size,
        window_step=window_step
    )
    # we'll use records based on overlapping windows of multiple paragraphs for embeddings
    records = records_windows

    if use_e5:
        model_name = E5_SMALL_NAME
        embs_func = _build_e5_embeddings(model_name, config.MODELS_DIR_PATH)
        embs_file_path = 'embs_e5.pkl'
    else:
        model_name = MPNET_NAME
        embs_file_path = 'embs_st.pkl'
        embs_func = _build_st_embeddings(model_name, config.MODELS_DIR_PATH, DEVICE_MPS)

    if not os.path.isfile(embs_file_path):
        print('calculating embeddings...')
        records_embs = sd.calculate_embeddings(
            records,
            lambda x: embs_func(x, query=False),
            batch_size=16
        )
        print('done')
        util.save_pkl(records_embs, embs_file_path)
        print(f'saved embeddings to `{embs_file_path}`')
    else:
        records_embs = util.load_pkl(embs_file_path)
        print(f'loaded embeddings from `{embs_file_path}`')

    print('total records:', len(records_embs))
    assert len(records) == len(records_embs)

    # ~~~~ load LLM ~~~~

    llm_model_file_path = os.path.join(
        config.MODELS_DIR_PATH,
        MISTRAL_7B_INSTRUCT_03
    )

    llama_model = _load_llama2(
        model_file_path=llm_model_file_path,
        context_tokens=MAX_CONTEXT_TOKENS,
        seed=seed
    )

    llama_model_wrapped = _build_llama2_wrapper(
        llama_model=llama_model,
        max_tokens=-1             # maximum generated tokens will be limited by context size
    )

    while True:

        search_str = input('Question > ')
        if not search_str:
            return 0

        # find single embedding of search string
        search_emb = embs_func([search_str], query=True)[0].cpu().numpy()

        # perform a very simple vector search
        _, idxs = _find_most_similar_idxs(records_embs, search_emb, MAX_RECORDS)

        print('~~~~ ' * 4)

        print('Context uids:')
        for idx in idxs:
            print(records_embs[idx]['uid'])

        print('~~~~ ' * 4)

        context = '\n\n'.join([
            '(CONTEXT SECTION)\n\n' + records_embs[idx]['text']
            for idx in idxs]
        )

        # TODO: trim to max input length

        prompt = qa_prompt(search_str, context)

        print(prompt)
        print('~~~~ ' * 4)

        answer_message = llama_model_wrapped(
            [dict(role='user', content=prompt)]
        )

        if answer_message is None:
            print('LLM inference failed!')
            continue

        answer = answer_message['content']
        answer = answer[1:]   # trim extra space at the beginning

        print()
        print('Answer:')
        print(answer)
        print('~~~~ ' * 4)


def _find_most_similar_idxs(
        records: List[Dict],
        embedding: np.ndarray,
        n: float
        ) -> Tuple[List[float], List[int]]:
    """
    Simplest vector search implementation.
    Linear search to find the minimum inner product.
    """
    print(embedding.__class__)

    scores = []
    for idx, record in tqdm.tqdm(enumerate(records)):
        score = np.sum(embedding * record['emb'])
        scores.append(score.item())

    idxs = list(reversed(list(np.argsort(scores)[-n:])))

    return (
        [scores[idx] for idx in idxs],
        idxs
    )


def _build_llama2_wrapper(
        llama_model: llama_cpp.Llama,
        max_tokens: int
        ) -> Callable:
    """
    Wrap a llama_cpp model to take a list of message dicts as input
    and return a single message dict.
    """

    def run(messages: List[Dict]) -> Optional[Dict]:
        """Run LLM chat completion."""
        try:
            output = llama_model.create_chat_completion(
                messages=messages,
                temperature=0.7,
                repeat_penalty=1.1,
                max_tokens=max_tokens
            )
            response = output['choices'][0]['message']
        except Exception as e:
            print('Exception:', str(e))
            response = None

        return response

    return run


def qa_prompt(question: str, context: str) -> str:
    """A very basic question-answering prompt."""
    return (
        'CONTEXT:\n\n' +
        context + '\n\n' +
        'QUESTION:\n\n' +
        question + '\n\n' +
        'TASK:\n\n' +
        'Use the above context and question, use information in the context to answer the question. ' +
        'Use ONLY information from the context! Some of the information in the context may be irrelevant. ' +
        'Keep your response brief and avoid chatting.'
    )


def _build_e5_embeddings(model_name: str, models_dir_path: str) -> Callable:
    """Build an embeddings function that using an e5 model."""

    tokenizer, model = emb.load_e5(model_name, models_dir_path)

    def embs_func(texts: List[str], query: bool) -> torch.Tensor:
        """function to get embeddings"""
        if not query:
            texts = ['passage: ' + x for x in texts]
        else:
            texts = ['query: ' + x for x in texts]
        return emb.e5_embeddings(tokenizer, model, texts)

    return embs_func


def _build_st_embeddings(model_name: str, models_dir_path: str, device: str) -> Callable:
    """Build an embeddings function using a SentenceTransformers model."""

    model = SentenceTransformer(model_name, cache_folder=models_dir_path, device=device)

    def call(texts: List[str], query: bool) -> torch.Tensor:
        embeddings = model.encode(
            texts,
            convert_to_numpy=False,
            convert_to_tensor=True
        )
        return embeddings

    return call


def _load_llama2(model_file_path, context_tokens: int, seed: int) -> llama_cpp.Llama:
    """Load llama 2 using llama-cpp-python"""
    res = llama_cpp.Llama(
        model_path=model_file_path,
        n_gpu_layers=10000,
        n_ctx=context_tokens,
        seed=seed,
        verbose=LLAMA_CPP_VERBOSE
    )
    return res


if __name__ == '__main__':
    main()
