from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class RerankerFeature(MindSpeedFeature):
    """Args for the generative reranker posttrain stage (`--stage reranker`)."""

    def __init__(self):
        super(RerankerFeature, self).__init__(feature_name="reranker", optimization_level=0)

    def validate_args(self, args):
        if args.stage != 'reranker':
            return
        if args.reranker_max_negatives < 1:
            raise ValueError('--reranker-max-negatives must be >= 1')
        if args.reranker_temperature <= 0:
            raise ValueError('--reranker-temperature must be positive')
        if args.reranker_positive_token == args.reranker_negative_token:
            raise ValueError('--reranker-positive-token and --reranker-negative-token '
                             'must differ')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--reranker-max-negatives', type=int, default=5,
                           help='Max negative docs per query group (actual group size is '
                                'variable: 1 pos + 1~N neg). Must match preprocess_data_'
                                'reranker.py --max-negatives.')
        group.add_argument('--reranker-loss-type', type=str, default='listwise',
                           choices=['listwise', 'pointwise'],
                           help='listwise: in-group softmax CE on the positive position '
                                '(= swift listwise_reranker); pointwise: per-sequence BCE '
                                'on the yes-no score (= swift pointwise_reranker, the '
                                'Qwen3-Reranker official SFT objective).')
        group.add_argument('--reranker-temperature', type=float, default=1.0,
                           help='Softmax temperature of the listwise reranker loss '
                                '(listwise only).')
        group.add_argument('--reranker-positive-token', type=str, default='yes',
                           help='Score token treated as the positive label. Must match '
                                'preprocess_data_reranker.py --positive-token. Env '
                                'GENERATIVE_RERANKER_POSITIVE_TOKEN overrides (ms-swift compat).')
        group.add_argument('--reranker-negative-token', type=str, default='no',
                           help='Score token treated as the negative label. Must match '
                                'preprocess_data_reranker.py --negative-token. Env '
                                'GENERATIVE_RERANKER_NEGATIVE_TOKEN overrides (ms-swift compat).')
