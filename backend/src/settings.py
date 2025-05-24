import os 

file_path = os.path.dirname( os.path.realpath( __file__ ) )

pretrained_models = {
    'googlenews': f'{file_path}/pretrained_models/GoogleNews-vectors-negative300.bin',
    'fasttext': f'{file_path}/pretrained_models/cc.en.300.bin',
    'glove': f'{file_path}/pretrained_models/glove.6B.300d.txt',
    'judge-model': f'{file_path}/pretrained_models/judge_model',
    'judge-model-tokenizer': f'{file_path}/pretrained_models/judge_model_tokenizer',
}
