import torch
from layout_generate.layoutPlanter import LayoutPlanter
from layout_generate.PosterO import PosterO
from sample_select.sampleRanker import SampleRanker
from utils import set_seed, get_args_infer_dataset
# from vllm import LLM, SamplingParams
import os

def main():
    set_seed()
    
    args = get_args_infer_dataset()
    
    if args.debug:
        print("[Debug mode] Load model? [y/n]", end=' ')
        if input().lower() == 'y':
            load_model = True
        else:
            load_model = False
    if os.path.exists(args.save_path.format('plain', 'top')):
        print(f"{args.save_path.format('plain', 'top')} already exists.")
        return None
    
    from vllm import LLM, SamplingParams
    if not args.debug or load_model:
        model = LLM(args.model_dir)
        sampl = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            n=args.num_return,
            stop=[args.stop_token],
        )
    else:
        model = None
        sampl = None

    if args.dataset_name in ['pku', 'cgl']:
        # rect only
        prompt_dict = {
            'opening': "The following are some scalable vector graphics (svg) allocating elements on the canvas.\n",
            'rag_opening': "Example {}: ",
            'rule': "First, learn from the examples and understand how this template works.\nThen, create a new one while following the rules:\n1. The svg must be meaningful, which implies that empty, all-zero, or symbolic attributes are not allowed.\n2. <rect> is the only legal svg tag, and the inner <rect> must be within the outer <svg>.\n3. The id of <rect> must be unique and picked from {}.\n4. The position of <rect> should be clustered neatly in avaliable areas while avoiding intersection. If intersected, <rect> should be resized or moved.\n",
            'pulse_appendix': ""
        }
    elif args.dataset_name.startswith('ps'):
        prompt_dict = {
            'opening': "The following are some scalable vector graphics (svg) allocating elements on the canvas.\n",
            'rag_opening': "Example {}: ",
            'rule': "First, learn from the examples and understand how this template works.\nThen, create a new one while following the rules:\n1. The svg must be meaningful, which implies that empty, all-zero, or symbolic attributes are not allowed.\n2. <rect>, <ellipse>, and <path> are the only legal svg tag, and the inner tags must be within the outer <svg>.\n3. The id of each tag must be unique and picked from {}.\n4. The position of each tag should be clustered neatly in avaliable areas while avoiding intersection. If intersected, the tags should be resized or moved.\n",
            'pulse_appendix': ""
        }
    else:
        raise ValueError(f'Invalid dataset_name: {args.dataset_name}, please provide `prompt_dict` directly.')
    
    if args.pool_strategy in ['metric_describe', 'metric_filter_describe']:
        if args.dataset_name == 'pku':
            avg_metric = {'ove': 0.0009740273,
                        'ali': 0.0037873797,
                        'und_l': 0.9941962509,
                        'und_s': 0.9902668416,
                        'uti': 0.2238318902,
                        'occ': 0.1192603177,
                        'rea': 0.0109249784,
                        'cov': 0.4414244997,
                        'con': 0.0995680822,}
        elif args.dataset_name == 'cgl':
            avg_metric = {'ove': 0.0002940352,
                        'ali': 0.0023839811,
                        'und_l': 0.9963430045,
                        'und_s': 0.9880051869,
                        'uti': 0.1984191850,
                        'occ': 0.1352170089,
                        'rea': 0.0118605712,
                        'cov': 0.4036413842,
                        'con': 0.0883146581,}
            
        metric_description_map = {'ove': 'Overlay',
                                'ali': 'Alignment',
                                'und_l': 'Underlay Effectiveness (Loose)',
                                'und_s': 'Underlay Effectiveness (Strict)',
                                'uti': 'Non-salient Region Utilization',
                                'occ': 'Salient Region Occlusion',
                                'rea': 'Non-readability',
                                'cov': 'Intention Region Coverage',
                                'con': 'Intention Region Conflict',}
        metric_description = ', '.join([f'{metric_description_map[k]}={v:.3f}' for k, v in avg_metric.items() if k in args.describe_list])
        prompt_dict['pulse_appendix'] = f'It achieves the following metrics: {metric_description}.\n'
    
    for s in args.structure:
        for i in args.injection:
            strategy = {
                'structure': s,
                'injection': i
            }
            layout_planter = LayoutPlanter(strategy, dataset_info=args.dataset_info, ext_save_name=args.ext_save_name, canvas_size=args.canvas_size)
            sampler = SampleRanker(args, layout_planter, args.pool_strategy, args.rank_strategy)
            posterO = PosterO(model, layout_planter, sampler)
            
            if os.path.exists(args.save_path.format(strategy['structure'], strategy['injection'])):
                print(f"{args.save_path.format(strategy['structure'], strategy['injection'])} already exists. Skip[0] or stop[1]?:", end=' ')
                ans = input()
                if ans == '0':
                    continue
                elif ans == '1':
                    return -1
            elif os.path.exists(args.save_path.format(strategy['structure'], strategy['injection']) + '.tmp'):
                print(f"{args.save_path.format(strategy['structure'], strategy['injection'])}.tmp exists. Loading...", end=' ')
                tmp = torch.load(args.save_path.format(strategy['structure'], strategy['injection']) + '.tmp', weights_only=False)
                layout_planter.db_valid = tmp['valid']
                layout_planter.db_test = tmp['test']
                checkpoint = tmp['checkpoint'] + 1
                if 'generated' in layout_planter.db_valid[checkpoint]:
                    checkpoint = {'valid': len(layout_planter.db_valid), 'test': checkpoint}
                else:
                    checkpoint = {'valid': checkpoint, 'test': 0}
                
                print("Done.")
            else:
                print("Temp file will be saved to", args.save_path.format(strategy['structure'], strategy['injection']) + '.tmp')
                checkpoint = {'valid': 0, 'test': 0}
                
            if args.debug:
                print("args:", args)
                print("strategy:", strategy)
                print("checkpoint:", checkpoint)
                checkpoint = {'valid': 0, 'test': 9}
                posterO.debug_from_planter(sampl, prompt_dict, args.dataset_info['label_info'], args.label_rback, args.N, checkpoint=checkpoint)
            else:
                posterO.inference_from_planter(sampl, prompt_dict, args.dataset_info['label_info'], args.save_path.format(strategy['structure'], strategy['injection']), args.label_rback, args.N, checkpoint=checkpoint)
    
if __name__ == '__main__':
    main()