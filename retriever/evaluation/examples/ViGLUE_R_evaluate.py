from ..evaluator.rerank import RerankingEvaluator
from ..evaluate import JustEvaluate
from .models import MODEL_WRAPPERS

def preprocess(dataset):
    return {
        'embed': {
            'query': dataset['anchor'],
            'positive': dataset['pos'],
            'negative': dataset['neg'],
        },
        'labels': {}
    }

def evaluate(
        model_name='ContextSearchLM/viennilm_dropinfonce_prompt_finetune_triplet',
        model_type='mean_pooling',
        token=None,
        device='cuda',
        ):
    print(model_name)
    print('mnli-r')
    mnli_r = JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='ContextSearchLM/ViGLUE-R',
        evaluator=RerankingEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=preprocess,
        result_folder='./results',
        split='mnli_r',
    ).run()

    print('qnli-r')
    qnli_r = JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='ContextSearchLM/ViGLUE-R',
        evaluator=RerankingEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=preprocess,
        result_folder='./results',
        split='qnli_r',
    ).run()

    return mnli_r, qnli_r